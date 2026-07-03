"""
main.py
=======
AI Civic Assistant Agent - Indian Pension Scheme Advisor
Entry point.  Run with:   python main.py

Flow
----
  1.  Collect user profile from terminal (with input validation)
  2.  Ask for optional personal documents folder (for auto-fill)
  3.  Search + LLM -> top 3 pension scheme recommendations
  4.  User picks one scheme
  5.  RAG pipeline indexes the application form PDF
  6.  User documents are indexed for auto-fill
  7.  Form fields are auto-filled where possible; user types the rest
  8.  Filled PDF is written to disk
"""

import os
import sys

from scheme_search    import get_pension_recommendations
from rag_form_reader  import load_and_index_form, extract_fields_from_rag
from user_docs_reader import build_user_doc_index, autofill_fields
from form_filler      import collect_user_inputs, fill_and_save_pdf

# Scheme name  ->  local form PDF path
SCHEME_PDF_MAP = {
    "atal pension yojana"           : os.path.join("forms", "atal_form.pdf"),
    "atal pension"                  : os.path.join("forms", "atal_form.pdf"),
    "national pension system"       : os.path.join("forms", "nps_form.pdf"),
    "nps"                           : os.path.join("forms", "nps_form.pdf"),
    "senior citizens savings scheme": os.path.join("forms", "scss_form.pdf"),
    "scss"                          : os.path.join("forms", "scss_form.pdf"),
    "employees provident fund"      : os.path.join("forms", "epf_form.pdf"),
    "epf"                           : os.path.join("forms", "epf_form.pdf"),
    "pm shram yogi"                 : os.path.join("forms", "pmsym_form.pdf"),
    "pmsym"                         : os.path.join("forms", "pmsym_form.pdf"),
}

# Input helpers  (with validation so bad typing doesn't crash the program)
def _input_int(prompt, min_val=0, max_val=150):
    """Keep asking until the user types a valid integer."""
    while True:
        raw = input(prompt).strip()
        try:
            val = int(raw)
            if min_val <= val <= max_val:
                return val
            print(f"  Please enter a number between {min_val} and {max_val}.")
        except ValueError:
            print("  Please enter a whole number (e.g. 35).")


def _input_float(prompt):
    """Keep asking until the user types a valid positive number."""
    while True:
        raw = input(prompt).strip()
        try:
            val = float(raw)
            if val >= 0:
                return val
            print("  Please enter a positive number.")
        except ValueError:
            print("  Please enter a number (e.g. 5.5).")


def _input_nonempty(prompt):
    """Keep asking until the user types something."""
    while True:
        raw = input(prompt).strip()
        if raw:
            return raw
        print("  This field cannot be empty.")

# Profile collection

def collect_profile():
    print("\n" + "=" * 60)
    print("  AI Civic Assistant  --  Indian Pension Scheme Advisor")
    print("=" * 60)
    print("Please fill in your details.  All fields are required.\n")

    profile = {}
    profile["name"]       = _input_nonempty("Full Name           : ")
    profile["age"]        = _input_int("Age                 : ", 1, 120)
    profile["gender"]     = _input_nonempty("Gender (M/F/Other)  : ")
    profile["occupation"] = _input_nonempty("Occupation          : ")
    profile["income_lpa"] = _input_float("Annual Income (LPA) : ")
    profile["state"]      = _input_nonempty("State               : ")
    return profile

# Documents folder prompt

def ask_docs_folder():
    print()
    print("-" * 60)
    print("  DOCUMENT AUTO-FILL  (optional but recommended)")
    print()
    print("  If you have scanned copies of your Aadhaar, PAN card,")
    print("  bank passbook, birth certificate, or salary slip, put")
    print("  them all in one folder.  The agent will read them and")
    print("  automatically fill matching fields in the form.")
    print()
    print("  Supported formats: PDF, PNG, JPG, JPEG, TIFF, BMP, WEBP")
    print("-" * 60)
    folder = input("\nPath to documents folder (press Enter to skip): ").strip()
    return folder if folder else None

# Scheme -> PDF resolution

def resolve_form_pdf(scheme_name):
    """Return the local PDF path for the chosen scheme, or None."""
    lower = scheme_name.lower()
    for key, path in SCHEME_PDF_MAP.items():
        if key in lower:
            return path
    return None

# Main flow

def main():
    # 1. Profile
    profile = collect_profile()

    # 2. Documents folder (optional)
    docs_folder = ask_docs_folder()

    # 3. Search + LLM recommendation
    print("\n" + "=" * 60)
    print("  SEARCHING FOR PENSION SCHEMES ...")
    print("=" * 60)
    schemes = get_pension_recommendations(profile)

    if not schemes:
        print()
        print("ERROR: Could not generate scheme recommendations.")
        print("  - Make sure Ollama is running:  ollama serve")
        print("  - Make sure the model is pulled:  ollama pull llama3:8b")
        print("  - Check your internet connection (needed for DuckDuckGo search)")
        sys.exit(1)

    # 4. Display options
    print()
    print("=" * 60)
    print("  TOP 3 RECOMMENDED PENSION SCHEMES FOR YOU")
    print("=" * 60)

    for i, s in enumerate(schemes, 1):
        print()
        print(f"  Option {i}: {s.get('name', 'Unknown Scheme')}")
        print(f"  {'Eligibility':<14}: {s.get('eligibility', 'N/A')}")
        print(f"  {'Pros':<14}: {s.get('pros', 'N/A')}")
        print(f"  {'Cons':<14}: {s.get('cons', 'N/A')}")
        print(f"  {'Documents':<14}: {s.get('documents', 'N/A')}")
        print(f"  {'Where to Apply':<14}: {s.get('where_to_apply', 'N/A')}")
        print(f"  {'How to Apply':<14}: {s.get('application_process', 'N/A')}")

    # 5. User picks a scheme
    print()
    while True:
        try:
            choice = int(input("Enter your choice (1 / 2 / 3): ").strip())
            if 1 <= choice <= len(schemes):
                break
            print(f"  Please enter a number between 1 and {len(schemes)}.")
        except ValueError:
            print("  Please type 1, 2, or 3.")

    selected = schemes[choice - 1]
    scheme_name = selected.get("name", f"Scheme_{choice}")
    print(f"\nSelected: {scheme_name}")

    # 6. Locate the form PDF
    pdf_path = resolve_form_pdf(scheme_name)

    if not pdf_path:
        print(f"\nNo pre-mapped PDF found for '{scheme_name}'.")
        pdf_path = input(
            "Enter the full path to the form PDF, or press Enter to exit: "
        ).strip()
        if not pdf_path:
            print("No PDF provided.  Exiting.")
            sys.exit(0)

    if not os.path.exists(pdf_path):
        print(f"\nERROR: File not found: {pdf_path}")
        print("  Place the PDF in the forms/ folder or provide the correct path.")
        sys.exit(1)

    # 7. RAG: index the application form and extract fields
    print()
    print("=" * 60)
    print("  READING APPLICATION FORM (RAG PIPELINE)")
    print("=" * 60)

    try:
        form_retriever = load_and_index_form(pdf_path)
    except Exception as exc:
        print(f"ERROR loading form PDF: {exc}")
        sys.exit(1)

    fields = extract_fields_from_rag(form_retriever, scheme_name)
    if not fields:
        print("ERROR: Could not extract any fields from the form.")
        print("  Try a cleaner PDF, or check that Ollama is responding.")
        sys.exit(1)

    print(f"\nExtracted {len(fields)} field(s) from the form.")

    # 8. Index user documents (optional)
    user_retriever = None
    if docs_folder:
        print()
        print("=" * 60)
        print("  INDEXING YOUR PERSONAL DOCUMENTS (RAG PIPELINE)")
        print("=" * 60)
        user_retriever = build_user_doc_index(docs_folder)
        # user_retriever may still be None if the folder was empty or unreadable

    # 9. Auto-fill from user documents via similarity search
    auto_filled = autofill_fields(fields, user_retriever)

    # 10. Interactive field collection (skip auto-filled, prompt for the rest)
    user_data = collect_user_inputs(fields, profile, auto_filled)

    # 11. Write the filled PDF
    fill_and_save_pdf(pdf_path, user_data, scheme_name, auto_filled)


if __name__ == "__main__":
    main()
