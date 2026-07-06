# AI Civic Assistant — Indian Pension Scheme Advisor

A terminal-based agent that helps Indian citizens discover the pension scheme that fits them, then auto-fills the actual application form using their own documents.

It combines live web search, local LLM reasoning (via Ollama), and two Retrieval-Augmented Generation (RAG) pipelines — one to read the application form, one to read the user's personal documents — so that as much of the form as possible gets filled automatically.

## How it works

1. **Collect profile** — name, age, gender, occupation, income, state (typed in the terminal, with input validation).
2. **Recommend schemes** — searches DuckDuckGo for scheme info relevant to the profile, then asks a local LLM (`llama3:8b` via Ollama) to return the top 3 matches with eligibility, pros/cons, required documents, and how to apply.
3. **Pick a scheme** — the user selects one of the three recommendations.
4. **Read the form (RAG)** — the matching application form PDF is loaded, OCR'd if needed, chunked, embedded, and indexed in FAISS. The LLM then extracts every field the applicant must fill.
5. **Read personal documents (RAG, optional)** — if the user points to a folder of scanned documents (Aadhaar, PAN, passbook, etc.), those are OCR'd, chunked, embedded, and indexed the same way.
6. **Auto-fill** — each form field is matched against the user-document index; the LLM extracts a value if it can find one with confidence.
7. **Fill remaining fields** — for anything not auto-filled, the profile is checked next (name, age, state, etc.), then the user is prompted directly, with a suggested value they can accept or overwrite.
8. **Generate the filled PDF** — writes directly into the form's AcroForm fields if it has any; otherwise generates a clean, color-coded PDF from scratch with ReportLab (green = from documents, blue = from profile, white = typed manually).

## Project structure

```
.
├── main.py               # Entry point — orchestrates the full flow
├── scheme_search.py      # DuckDuckGo search + LLM -> top 3 scheme recommendations
├── rag_form_reader.py    # RAG pipeline for the application form PDF
├── user_docs_reader.py   # RAG pipeline for the user's personal documents
├── form_filler.py        # Field collection + PDF generation (AcroForm or ReportLab)
├── requirements.txt
└── forms/                # Local PDF forms, keyed by scheme name in main.py
    ├── atal_form.pdf
    ├── nps_form.pdf
    ├── scss_form.pdf
    ├── epf_form.pdf
    └── pmsym_form.pdf
```

## Requirements

- Python 3.9+
- [Ollama](https://ollama.com) running locally, with two models pulled:
  ```bash
  ollama pull llama3:8b
  ollama pull nomic-embed-text
  ```
- Tesseract OCR binary (for scanned PDFs and image documents):

  | OS | Install |
  |---|---|
  | Windows | [UB-Mannheim build](https://github.com/UB-Mannheim/tesseract/wiki) |
  | Ubuntu | `sudo apt install tesseract-ocr poppler-utils` |
  | macOS | `brew install tesseract poppler` |

- Internet connection (for the DuckDuckGo scheme search)

## Setup

```bash
# 1. Clone and enter the project
git clone <your-repo-url>
cd <your-repo-folder>

# 2. Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Start Ollama and pull the models (in a separate terminal)
ollama serve
ollama pull llama3:8b
ollama pull nomic-embed-text

# 5. Add the government form PDFs to forms/
#    (see SCHEME_PDF_MAP in main.py for the expected filenames)
```

## Usage

```bash
python main.py
```

You'll be walked through:

1. Entering your profile (name, age, gender, occupation, income, state).
2. Optionally pointing to a folder of scanned documents for auto-fill (PDF, PNG, JPG, JPEG, TIFF, BMP, WEBP).
3. Reviewing 3 recommended pension schemes and picking one.
4. Watching the agent read the form and your documents, then auto-fill what it can.
5. Confirming or typing in any remaining fields.

The filled PDF is saved to the current directory as `filled_<scheme_name>.pdf`.

## Supported schemes (out of the box)

- Atal Pension Yojana (APY)
- National Pension System (NPS)
- Senior Citizens Savings Scheme (SCSS)
- Employees' Provident Fund (EPF)
- PM Shram Yogi Maan-dhan (PM-SYM)

Add more by extending `SCHEME_PDF_MAP` in `main.py` and dropping the corresponding form PDF into `forms/`.

## Notes & troubleshooting

- **"Could not generate scheme recommendations"** — make sure `ollama serve` is running, `llama3:8b` is pulled, and you have an internet connection for the DuckDuckGo search.
- **Scanned/image-based PDFs** are handled automatically via OCR fallback — no extra steps needed, as long as Tesseract is installed.
- **Fields not found automatically** are never guessed — the agent only fills a field from your documents if it's confident, otherwise it asks you directly.
- Chunk sizes are tuned per pipeline: form reading uses 800/100 (chunk size/overlap), personal documents use 600/80.

## Disclaimer

This tool is an informational aid, not official government guidance. Always verify scheme eligibility and form details on the official government portal or with a bank/post office before submitting an application.
