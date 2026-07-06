"""
rag_form_reader.py
==================
Loads a government pension application form PDF into a FAISS vector index,
then uses similarity search + LLM to extract every field the applicant must fill.

RAG pipeline
------------
  PDF  ->  text extraction  ->  chunking  ->  embeddings  ->  FAISS index
                                                                    |
  query: "form fields applicant name ..."  ->  top-k chunks  ->  LLM
                                                                    |
                                                         list of field dicts

PDF loading strategy
--------------------
  1.  PyPDFLoader  - works for digital / text-layer PDFs
  2.  pytesseract OCR via pdf2image  - fallback for scanned / image PDFs
      (triggers automatically when PyPDF returns fewer than 50 characters)

Exported
--------
  load_and_index_form(pdf_path) -> retriever
  extract_fields_from_rag(retriever, scheme_name) -> list[dict]
"""

import re
import json
from pathlib import Path

# Text splitter  (import path changed between LangChain versions)
try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    try:
        from langchain.text_splitter import RecursiveCharacterTextSplitter
    except ImportError:
        raise ImportError(
            "RecursiveCharacterTextSplitter not found.\n"
            "Install with:  pip install langchain-text-splitters"
        )

# OllamaEmbeddings  (moved to langchain_ollama in newer releases)
try:
    from langchain_ollama import OllamaEmbeddings
except ImportError:
    try:
        from langchain_community.embeddings import OllamaEmbeddings
    except ImportError:
        raise ImportError(
            "OllamaEmbeddings not found.\n"
            "Install with:  pip install langchain-ollama"
        )

# Document schema  (moved to langchain_core in newer releases)
try:
    from langchain_core.documents import Document
except ImportError:
    from langchain.schema import Document

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores     import FAISS
from langchain_ollama                     import OllamaLLM


# Shared instances
# Chunk size 800 / overlap 100 preserved exactly as requested.
llm        = OllamaLLM(model="llama3:8b", temperature=0)
embeddings = OllamaEmbeddings(model="llama3:8b")
splitter   = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)

# PDF loading
def _load_pypdf(path):
    """
    Load a PDF with PyPDFLoader.
    Returns a list of Document objects (one per page), or [] on failure.
    """
    try:
        pages = PyPDFLoader(str(path)).load()
        return pages
    except Exception as exc:
        print(f"  [PyPDF] Could not load {path.name}: {exc}")
        return []


def _load_ocr(path):
    """
    Convert each page to an image (pdf2image) and OCR it (pytesseract).
    Returns a list of Document objects, or [] if the libraries are missing.
    """
    try:
        import pytesseract
        from pdf2image import convert_from_path
    except ImportError as exc:
        print(
            f"  [OCR] Library missing: {exc}\n"
            "  Install:  pip install pytesseract pdf2image Pillow\n"
            "  Also install Tesseract binary  (see README Section 11)."
        )
        return []

    try:
        print("  [OCR] Rasterising PDF pages at 200 DPI ...")
        images = convert_from_path(str(path), dpi=200)
        docs   = []
        for i, img in enumerate(images):
            text = pytesseract.image_to_string(img)
            docs.append(Document(
                page_content=text,
                metadata={"page": i + 1, "source": str(path)}
            ))
        return docs
    except Exception as exc:
        print(f"  [OCR] Failed on {path.name}: {exc}")
        return []


def load_pdf_documents(pdf_path):
    """
    Load text from a PDF using PyPDF, falling back to OCR automatically.

    Returns a list of Document objects.
    Raises FileNotFoundError if the file does not exist.
    Raises ValueError if no text could be extracted by any method.
    """
    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Form PDF not found: {pdf_path}\n"
            "Place it in the forms/ folder and update SCHEME_PDF_MAP in main.py."
        )

    print(f"  Loading: {path.name}")
    docs  = _load_pypdf(path)
    total = sum(len(d.page_content) for d in docs)

    if total < 50:
        print(f"  [info] PyPDF extracted only {total} chars.  Switching to OCR ...")
        docs  = _load_ocr(path)
        total = sum(len(d.page_content) for d in docs)

    if not docs or total < 10:
        raise ValueError(
            f"Could not extract readable text from {pdf_path}.\n"
            "Check that the file is a valid PDF and not password-protected."
        )

    print(f"  {len(docs)} page(s) loaded, {total} characters extracted.")
    return docs

# FAISS vector index  (chunk size 800, overlap 100 - unchanged)

def build_faiss_index(docs):
    """
    Split documents into overlapping chunks, embed with OllamaEmbeddings,
    and store in an in-memory FAISS index.

    chunk_size=800, chunk_overlap=100  --  these are NOT changed.

    Returns a LangChain retriever configured for k=6.
    """
    print("  Splitting text into chunks (size=800, overlap=100) ...")
    chunks = splitter.split_documents(docs)
    print(f"  {len(chunks)} chunks created.")
    print("  Embedding with OllamaEmbeddings (llama3:8b) ...")
    print("  Note: first run may take 30-90 seconds on CPU.")

    try:
        vectorstore = FAISS.from_documents(chunks, embeddings)
    except Exception as exc:
        raise RuntimeError(
            f"FAISS indexing failed: {exc}\n"
            "Make sure Ollama is running and llama3:8b is available."
        )

    retriever = vectorstore.as_retriever(search_kwargs={"k": 6})
    print("  FAISS index ready.  (k=6 chunks per query)")
    return retriever


def load_and_index_form(pdf_path):
    """
    Public entry point.
    Loads the form PDF and returns a FAISS retriever.
    """
    docs = load_pdf_documents(pdf_path)
    return build_faiss_index(docs)

# Field extraction via RAG + LLM

def _retrieve(retriever, query):
    """Run a similarity query and join the top-k chunks into one string."""
    chunks = retriever.invoke(query)
    return "\n\n".join(c.page_content for c in chunks)


def _parse_fields(raw):
    """
    Parse the LLM's raw text into a list of field dicts.

    Attempt 1: extract the outermost JSON array (GREEDY so it captures
               the full multi-line array, not just the first bracket pair).
    Attempt 2: line-by-line reconstruction for prose-style responses.

    Each returned dict has keys: field_name, description, example.
    """
    # JSON array (greedy DOTALL) 
    match = re.search(r"\[.*\]", raw, re.DOTALL)
    if match:
        try:
            items = json.loads(match.group())
            result = []
            for item in items:
                if not isinstance(item, dict):
                    continue
                result.append({
                    "field_name" : (item.get("field_name") or
                                    item.get("name") or "").strip(),
                    "description": (item.get("description") or
                                    item.get("label") or "").strip(),
                    "example"    : item.get("example", "").strip(),
                })
            # Filter out empty entries
            result = [f for f in result if f["field_name"]]
            if result:
                return result
        except (json.JSONDecodeError, ValueError):
            pass

    #  line-by-line fallback 
    fields = []
    for line in raw.splitlines():
        line = line.strip(" -•*1234567890.)>")
        if len(line) < 4:
            continue
        parts = re.split(r"[:\|]", line, maxsplit=1)
        name  = parts[0].strip()
        desc  = parts[1].strip() if len(parts) > 1 else ""
        # Skip lines that look like prose rather than field names
        if len(name.split()) > 8:
            continue
        fields.append({"field_name": name, "description": desc, "example": ""})

    return fields


def extract_fields_from_rag(retriever, scheme_name):
    """
    Use the FAISS retriever to fetch the most relevant chunks from the form,
    then ask the LLM to list every field the applicant must fill.

    Returns a list of dicts: [{field_name, description, example}, ...]
    Returns [] if the LLM fails or returns unparseable output.
    """
    # Retrieve chunks most relevant to "form fields"
    context = _retrieve(
        retriever,
        "form fields applicant name date signature address bank account "
        "nomination blank lines boxes to fill in"
    )

    prompt = (
        f'You are reading a government pension application form for "{scheme_name}".\n\n'
        "FORM CONTENT (extracted from PDF):\n"
        f"{context[:3500]}\n\n"
        "TASK:\n"
        "Identify ALL fields the applicant must fill in this form.\n"
        "Do NOT hardcode anything.  Extract only what you see above.\n\n"
        "Return ONLY a valid JSON array.  No markdown, no explanation.\n"
        "Each object must have these 3 keys:\n"
        '  "field_name"  : short label for the field\n'
        '  "description" : what information goes here\n'
        '  "example"     : a realistic example value\n\n'
        "Cover all sections: personal details, address, bank details,\n"
        "nomination, declaration, signature, date.\n\n"
        "[\n"
        '  {"field_name": "Full Name", '
        '"description": "Applicant full name as per Aadhaar", '
        '"example": "Ramesh Kumar"},\n'
        "  ...\n"
        "]\n\n"
        "JSON array:"
    )

    print("  Querying FAISS index and asking LLM to extract fields ...")
    try:
        raw = llm.invoke(prompt)
    except Exception as exc:
        print(f"  [LLM error] {exc}")
        return []

    fields = _parse_fields(raw)

    if not fields:
        print("  [warning] Could not extract fields.  Raw LLM output:")
        print("  " + raw[:600].replace("\n", "\n  "))
    else:
        print(f"  {len(fields)} field(s) identified.")

    return fields
