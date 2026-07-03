"""
user_docs_reader.py
===================
Reads every PDF and image in the user's personal documents folder,
extracts text via OCR where needed, builds a FAISS vector index,
then uses similarity search + LLM to auto-fill form fields.

How it works
------------
  build_user_doc_index(folder)
      1.  Scan folder for supported files
      2.  Extract text from each file:
              PDF   -> PyPDFLoader  (falls back to pytesseract OCR if sparse)
              Image -> pytesseract OCR via Pillow
      3.  Chunk text  (size=600, overlap=80  -- kept as-is)
      4.  Embed with OllamaEmbeddings (llama3:8b, local)
      5.  Build FAISS in-memory index
      Returns retriever  or  None if no readable files found.

  autofill_fields(fields, retriever)
      For each form field:
          query  = field_name + description
          chunks = top-5 from FAISS similarity search
          value  = LLM extracts exact value from chunks  (or NOT_FOUND)
      Returns  {field_name: value_or_None}
      None  -> not found in documents; form_filler will ask user manually.

Supported file types
--------------------
  PDF   : .pdf
  Image : .png  .jpg  .jpeg  .tiff  .tif  .bmp  .webp

OCR requirements
----------------
  pip install pytesseract pdf2image Pillow
  Plus the Tesseract binary (see README Section 11):
    Windows : https://github.com/UB-Mannheim/tesseract/wiki
    Ubuntu  : sudo apt install tesseract-ocr poppler-utils
    macOS   : brew install tesseract poppler

Exported
--------
  build_user_doc_index(folder_path) -> retriever | None
  autofill_fields(fields, retriever) -> dict
"""

from pathlib import Path

# Text splitter
try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    try:
        from langchain.text_splitter import RecursiveCharacterTextSplitter
    except ImportError:
        raise ImportError(
            "RecursiveCharacterTextSplitter not found.\n"
            "Install:  pip install langchain-text-splitters"
        )

# OllamaEmbeddings
try:
    from langchain_ollama import OllamaEmbeddings
except ImportError:
    try:
        from langchain_community.embeddings import OllamaEmbeddings
    except ImportError:
        raise ImportError(
            "OllamaEmbeddings not found.\n"
            "Install:  pip install langchain-ollama"
        )

# Document schema
try:
    from langchain_core.documents import Document
except ImportError:
    from langchain.schema import Document

from langchain_community.vectorstores import FAISS
from langchain_ollama                  import OllamaLLM

# Shared instances  (same model as the rest of the pipeline)
# Chunk size 600, overlap 80 kept as-is.
llm        = OllamaLLM(model="llama3:8b", temperature=0)
embeddings = OllamaEmbeddings(model="llama3:8b")
splitter   = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=80)

PDF_EXTS   = {".pdf"}
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp", ".webp"}
# Text extraction helpers
def _text_from_pdf(path):
    """
    Extract text from a PDF.
    Strategy: PyPDFLoader first -> pytesseract OCR fallback.
    Returns plain text string (may be empty on failure).
    """
    # -- PyPDF --
    try:
        from langchain_community.document_loaders import PyPDFLoader
        pages = PyPDFLoader(str(path)).load()
        text  = " ".join(p.page_content for p in pages).strip()
        if len(text) >= 50:
            return text
        # If we got some text but very little, still try OCR
        if text:
            print(f"      PyPDF returned only {len(text)} chars, also trying OCR ...")
    except Exception as exc:
        print(f"      [PyPDF] {path.name}: {exc}")

    # -- OCR fallback --
    try:
        import pytesseract
        from pdf2image import convert_from_path

        images = convert_from_path(str(path), dpi=200)
        parts  = []
        for img in images:
            parts.append(pytesseract.image_to_string(img))
        return "\n".join(parts).strip()

    except ImportError:
        print(
            "      [OCR] pytesseract / pdf2image not installed.\n"
            "      Install:  pip install pytesseract pdf2image Pillow\n"
            "      And the Tesseract binary (see README)."
        )
        return ""
    except Exception as exc:
        print(f"      [OCR-PDF] {path.name}: {exc}")
        return ""


def _text_from_image(path):
    """
    Extract text from an image file using pytesseract OCR.
    Returns plain text string (may be empty on failure).
    """
    try:
        import pytesseract
        from PIL import Image

        # Convert to RGB to handle TIFF, BMP, WEBP, CMYK, etc.
        img  = Image.open(str(path)).convert("RGB")
        text = pytesseract.image_to_string(img)
        return text.strip()

    except ImportError:
        print(
            f"      [OCR] Cannot read image {path.name}.\n"
            "      Install:  pip install pytesseract Pillow\n"
            "      And the Tesseract binary (see README)."
        )
        return ""
    except Exception as exc:
        print(f"      [OCR-image] {path.name}: {exc}")
        return ""

# FAISS index builder  (chunk size 600, overlap 80 unchanged)

def build_user_doc_index(folder_path):
    """
    Scan folder_path for supported files, extract text, embed, and index.

    Returns a FAISS retriever configured for k=5, or None if:
      - folder does not exist
      - no supported files found
      - no text could be extracted from any file
    """
    folder = Path(folder_path)

    if not folder.exists():
        print(f"  [user-docs] Folder not found: {folder_path}")
        return None

    if not folder.is_dir():
        print(f"  [user-docs] Path is not a directory: {folder_path}")
        return None

    supported = PDF_EXTS | IMAGE_EXTS
    files = sorted(
        [f for f in folder.iterdir()
         if f.is_file() and f.suffix.lower() in supported]
    )

    if not files:
        print(
            f"  [user-docs] No supported files found in: {folder_path}\n"
            f"  Supported types: {', '.join(sorted(supported))}"
        )
        return None

    print(f"\n  Found {len(files)} document(s) in {folder_path}")
    print("  Extracting text (OCR used for images and scanned PDFs) ...")

    docs = []
    for f in files:
        print(f"\n    Reading: {f.name}")
        if f.suffix.lower() in PDF_EXTS:
            text = _text_from_pdf(f)
        else:
            text = _text_from_image(f)

        if text:
            docs.append(Document(
                page_content=text,
                metadata={"source": f.name, "type": f.suffix.lower()}
            ))
            print(f"      OK  ->  {len(text)} characters extracted.")
        else:
            print(f"      SKIP  ->  No text could be extracted.")

    if not docs:
        print(
            "\n  [user-docs] Could not extract text from any document.\n"
            "  Check that Tesseract is installed and the files are readable."
        )
        return None

    # Build FAISS index
    print(f"\n  Building FAISS index from {len(docs)} document(s) ...")
    print("  Chunking (size=600, overlap=80) ...")
    chunks = splitter.split_documents(docs)
    print(f"  {len(chunks)} chunks created.")
    print("  Embedding with OllamaEmbeddings (llama3:8b) ...")

    try:
        vectorstore = FAISS.from_documents(chunks, embeddings)
    except Exception as exc:
        print(f"  [FAISS] Indexing failed: {exc}")
        return None

    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    print("  User document index ready.  (k=5 chunks per query)")
    return retriever

# Similarity search + LLM extraction
def _retrieve_for_field(retriever, field_name, description):
    """
    Query the user-document FAISS index with the field name and description.
    Returns the top-5 most relevant chunks joined as a single string.
    """
    query  = f"{field_name} {description}".strip()
    chunks = retriever.invoke(query)
    return "\n\n".join(c.page_content for c in chunks)


def _extract_value(field_name, description, context):
    """
    Ask the LLM to extract the exact value of field_name from context.

    Returns the value string, or None if:
      - LLM says NOT_FOUND
      - Response is empty
      - Response is suspiciously long (LLM went off-script, > 300 chars)
    """
    prompt = (
        "You are reading a person's personal identity documents.\n\n"
        f'Find the value for: "{field_name}"\n'
        f'Description: "{description}"\n\n'
        "DOCUMENT TEXT:\n"
        f"{context[:2500]}\n\n"
        "Rules:\n"
        "  - Reply with ONLY the value, nothing else.\n"
        "  - If the value is clearly present, return it exactly as written.\n"
        "  - If you cannot find it with confidence, reply: NOT_FOUND\n"
        "  - Do NOT guess, invent, or make up any value.\n"
        "  - Do NOT include the field name in your reply.\n\n"
        "Value:"
    )

    try:
        raw = llm.invoke(prompt).strip()

        # Strip surrounding quotes the LLM sometimes adds
        raw = raw.strip('"\'')

        # Reject empty or explicit not-found
        if not raw or raw.upper() in ("NOT_FOUND", "NOTFOUND", "N/A", "NA"):
            return None

        # Reject suspiciously long responses (LLM wrote prose instead of a value)
        if len(raw) > 300:
            return None

        # Reject multi-line responses (should be a single field value)
        if "\n" in raw and len(raw) > 100:
            return None

        return raw

    except Exception as exc:
        print(f"      [LLM error] {exc}")
        return None


def autofill_fields(fields, retriever):
    """
    For every form field, run RAG similarity search against the user-document
    index and ask the LLM to extract the value.

    Parameters
    ----------
    fields    : list of dicts from rag_form_reader.extract_fields_from_rag()
    retriever : FAISS retriever from build_user_doc_index(), or None

    Returns
    -------
    dict  {field_name: value_or_None}
      value  -> found in documents, will be suggested as auto-fill
      None   -> not found, form_filler will prompt the user manually
    """
    # If no user documents were indexed, return all None immediately
    if retriever is None:
        return {f.get("field_name", f"Field_{i}"): None
                for i, f in enumerate(fields, 1)}

    print()
    print("  Running RAG auto-fill:")
    print("  (For each field: similarity search -> LLM extraction)")
    print()

    results = {}
    total   = len(fields)

    for i, field in enumerate(fields, 1):
        name = field.get("field_name", f"Field_{i}")
        desc = field.get("description", "")

        # Similarity search in user document index
        context = _retrieve_for_field(retriever, name, desc)

        # LLM extraction
        value = _extract_value(name, desc, context)

        # Print status
        if value:
            display = value[:45] + "..." if len(value) > 45 else value
            status  = f"=> {display}"
        else:
            status = "=> (not found in documents)"

        print(f"    [{i:>2}/{total}]  {name:<35}  {status}")

        results[name] = value

    # Summary
    found    = sum(1 for v in results.values() if v is not None)
    not_found = total - found
    print()
    print(f"  Auto-fill summary: {found} filled, {not_found} not found.")
    print(f"  Fields not found will be asked manually.")

    return results
