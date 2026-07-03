"""
form_filler.py
==============
Collects user answers for every form field (accepting auto-filled values
from scanned documents, or prompting manually when not found), then writes
a filled PDF using one of two strategies:

  Strategy 1 (preferred): Fill AcroForm fields in the original PDF  (pypdf)
  Strategy 2 (fallback) : Generate a clean new PDF with ReportLab

  In the ReportLab output:
    Green  rows = value auto-filled from the user's scanned documents
    Blue   rows = value taken from the user's profile (name, age, etc.)
    White  rows = value typed manually by the user

Exported
--------
  collect_user_inputs(fields, profile, auto_filled) -> dict
  fill_and_save_pdf(pdf_path, user_data, scheme_name, auto_filled)
"""

import os
from datetime import date

# Field collection
def collect_user_inputs(fields, profile, auto_filled=None):
    """
    Walk through every extracted form field and collect a value for each one.

    Value priority chain (highest to lowest):
      1.  auto_filled[field]   - found in user's scanned documents via RAG
      2.  profile_hints[field] - derived from the 6 terminal profile answers
      3.  field example hint   - from the form's field definition
      4.  empty                - user must type

    The user sees the suggested value and can press Enter to accept it or
    type something new to override it.

    Parameters
    ----------
    fields      : list of dicts [{field_name, description, example}, ...]
    profile     : dict from collect_profile() in main.py
    auto_filled : dict {field_name: value_or_None} from autofill_fields()

    Returns
    -------
    dict  {field_name: final_value}
    """
    if auto_filled is None:
        auto_filled = {}

    # Map lowercase common field names to known profile values
    today = date.today().strftime("%d/%m/%Y")
    profile_hints = {
        "name"              : profile.get("name", ""),
        "full name"         : profile.get("name", ""),
        "applicant name"    : profile.get("name", ""),
        "applicant's name"  : profile.get("name", ""),
        "subscriber name"   : profile.get("name", ""),
        "age"               : str(profile.get("age", "")),
        "age in years"      : str(profile.get("age", "")),
        "gender"            : profile.get("gender", ""),
        "sex"               : profile.get("gender", ""),
        "state"             : profile.get("state", ""),
        "state name"        : profile.get("state", ""),
        "place"             : profile.get("state", ""),
        "date"              : today,
        "today"             : today,
        "date of application": today,
        "application date"  : today,
        "occupation"        : profile.get("occupation", ""),
        "profession"        : profile.get("occupation", ""),
        "annual income"     : str(profile.get("income_lpa", "")),
        "income"            : str(profile.get("income_lpa", "")),
    }

    # Count sources
    auto_count    = sum(1 for v in auto_filled.values() if v is not None)
    profile_count = 0
    for f in fields:
        nm = f.get("field_name", "").lower()
        if auto_filled.get(f.get("field_name")) is None and nm in profile_hints:
            profile_count += 1
    manual_count  = len(fields) - auto_count - profile_count

    print()
    print("=" * 60)
    print("  FORM FIELD COLLECTION")
    print("=" * 60)
    print(f"  From your documents (auto)  : {auto_count}")
    print(f"  From your profile           : {profile_count}")
    print(f"  Need your input             : {manual_count}")
    print()
    print("  For each field:")
    print("    Press Enter    -> accept the suggested value")
    print("    Type + Enter   -> use your typed value instead")
    print("=" * 60)

    user_data = {}

    for i, field in enumerate(fields, 1):
        name = field.get("field_name", f"Field_{i}")
        desc = field.get("description", "")
        hint = field.get("example", "")

        # Build suggestion from priority chain
        doc_val     = auto_filled.get(name)
        profile_val = profile_hints.get(name.lower(), "")
        suggestion  = doc_val or profile_val or hint

        if doc_val:
            source_label = "[from your documents]"
        elif profile_val:
            source_label = "[from your profile]"
        elif hint:
            source_label = "[example]"
        else:
            source_label = ""

        # Display the prompt
        print()
        print(f"  [{i}/{len(fields)}]  {name}")
        if desc:
            print(f"           Info      : {desc}")
        if suggestion:
            print(f"           Suggested : {suggestion}  {source_label}")

        user_input = input("           Your answer: ").strip()

        # Accept suggestion on empty input
        if not user_input:
            if suggestion:
                user_input = suggestion
                print(f"           Accepted  : {user_input}")
            # else: leave empty (user chose not to fill)

        user_data[name] = user_input

    # Summary
    filled   = sum(1 for v in user_data.values() if v)
    unfilled = len(fields) - filled
    print()
    print(f"  Collection complete.  Filled: {filled}  |  Left blank: {unfilled}")
    return user_data

# PDF output helpers

def _is_fillable(pdf_path):
    """
    Return True if the PDF contains AcroForm fields.
    Returns False silently on any error (treats as not fillable).
    """
    try:
        import pypdf
        reader = pypdf.PdfReader(pdf_path)
        fields = reader.get_fields()
        return bool(fields)
    except Exception:
        return False


def _fill_acroform(pdf_path, user_data, output_path):
    """
    Write user_data values into the existing AcroForm fields of a PDF.

    Matching is done in two passes:
      1. Exact match  (case-insensitive, spaces removed)
      2. Prefix match (first 6 characters, case-insensitive)
    """
    import pypdf

    reader = pypdf.PdfReader(pdf_path)
    writer = pypdf.PdfWriter()

    for page in reader.pages:
        writer.add_page(page)

    writer.clone_reader_document_root(reader)

    pdf_fields = reader.get_fields() or {}
    mapping    = {}

    for pdf_key in pdf_fields:
        norm_pdf = pdf_key.lower().replace(" ", "").replace("_", "")

        # Pass 1: exact normalised match
        matched = False
        for data_key, val in user_data.items():
            norm_data = data_key.lower().replace(" ", "").replace("_", "")
            if norm_data == norm_pdf:
                mapping[pdf_key] = val
                matched = True
                break

        # Pass 2: prefix match (first 6 chars)
        if not matched:
            for data_key, val in user_data.items():
                if data_key.lower()[:6] in pdf_key.lower():
                    mapping[pdf_key] = val
                    break

    writer.update_page_form_field_values(writer.pages[0], mapping)

    with open(output_path, "wb") as fh:
        writer.write(fh)

    filled_count = sum(1 for v in mapping.values() if v)
    print(f"  AcroForm: {filled_count}/{len(pdf_fields)} fields written.")
    print(f"  Saved -> {output_path}")


def _generate_reportlab_pdf(user_data, scheme_name, auto_filled, profile, output_path):
    """
    Generate a clean structured PDF from scratch using ReportLab.

    Row colour coding:
      Green (#e8f5e9) = auto-filled from user's scanned documents
      Steel (#dce8f7) = derived from user profile
      White / light blue = typed manually
    """
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles    import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units     import mm
    from reportlab.lib           import colors
    from reportlab.lib.enums     import TA_CENTER
    from reportlab.platypus      import (
        SimpleDocTemplate, Paragraph, Spacer,
        Table, TableStyle, HRFlowable
    )

    S       = getSampleStyleSheet()
    title_s = ParagraphStyle("RL_title", parent=S["Title"],
                              fontSize=16, alignment=TA_CENTER, spaceAfter=4)
    sub_s   = ParagraphStyle("RL_sub",   parent=S["Normal"],
                              fontSize=10, textColor=colors.grey,
                              alignment=TA_CENTER)
    lbl_s   = ParagraphStyle("RL_lbl",   parent=S["Normal"],
                              fontSize=9,  textColor=colors.HexColor("#444444"),
                              leading=12)
    val_s   = ParagraphStyle("RL_val",   parent=S["Normal"],
                              fontSize=11, leading=14)
    grn_s   = ParagraphStyle("RL_grn",   parent=S["Normal"],
                              fontSize=8,  textColor=colors.HexColor("#1b5e20"),
                              leading=11)
    blu_s   = ParagraphStyle("RL_blu",   parent=S["Normal"],
                              fontSize=8,  textColor=colors.HexColor("#1a3a6b"),
                              leading=11)

    COL_HEADER = colors.HexColor("#1a73e8")
    COL_AUTO   = colors.HexColor("#e8f5e9")   # green  - from documents
    COL_PROF   = colors.HexColor("#dce8f7")   # steel  - from profile
    COL_ALT    = colors.HexColor("#f1f5ff")   # light blue - manual alt row
    COL_WHITE  = colors.white                 # manual normal row

    # Profile hint keys for source detection
    profile_keys = {
        "name", "full name", "applicant name", "age", "gender",
        "state", "place", "date", "today", "occupation",
        "profession", "annual income", "income",
        "subscriber name", "applicant's name", "age in years",
        "sex", "state name", "date of application", "application date",
    }

    doc   = SimpleDocTemplate(
        output_path, pagesize=A4,
        leftMargin=20*mm,  rightMargin=20*mm,
        topMargin=20*mm,   bottomMargin=20*mm
    )
    story = []

    # Header
    story.append(Paragraph("Application Form", title_s))
    story.append(Paragraph(scheme_name, sub_s))
    story.append(Spacer(1, 4*mm))
    story.append(HRFlowable(width="100%", thickness=1,
                             color=colors.HexColor("#1a73e8")))
    story.append(Spacer(1, 3*mm))

    # Colour legend
    legend_data = [[
        Paragraph("<b>Colour key:</b>", lbl_s),
        Paragraph("Green = from your documents", grn_s),
        Paragraph("Blue = from your profile",    blu_s),
        Paragraph("White = manual entry",         lbl_s),
    ]]
    leg_tbl = Table(legend_data, colWidths=[28*mm, 48*mm, 42*mm, 52*mm])
    leg_tbl.setStyle(TableStyle([
        ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING",    (0, 0), (-1, -1), 2),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
        ("LEFTPADDING",   (0, 0), (-1, -1), 4),
    ]))
    story.append(leg_tbl)
    story.append(Spacer(1, 5*mm))

    # Field/Value table
    header_row = [
        Paragraph("<b>Field</b>",  lbl_s),
        Paragraph("<b>Value</b>",  lbl_s),
        Paragraph("<b>Source</b>", lbl_s),
    ]
    rows   = [header_row]
    row_bgs = [COL_HEADER]

    for i, (fname, fval) in enumerate(user_data.items()):
        is_auto    = bool(auto_filled.get(fname))
        is_profile = (not is_auto) and (fname.lower() in profile_keys)

        if is_auto:
            source_text  = "Document scan"
            source_style = grn_s
            bg           = COL_AUTO
        elif is_profile:
            source_text  = "Profile"
            source_style = blu_s
            bg           = COL_PROF
        else:
            source_text  = "Manual"
            source_style = lbl_s
            bg           = COL_ALT if i % 2 == 0 else COL_WHITE

        rows.append([
            Paragraph(str(fname),             lbl_s),
            Paragraph(str(fval) if fval else "—", val_s),
            Paragraph(source_text,            source_style),
        ])
        row_bgs.append(bg)

    field_tbl = Table(rows, colWidths=[60*mm, 95*mm, 35*mm])

    # Build row-background commands separately then combine
    style_cmds = [
        ("BACKGROUND", (0, 0), (-1, 0), COL_HEADER),
        ("TEXTCOLOR",  (0, 0), (-1, 0), colors.white),
        ("GRID",          (0, 0), (-1, -1), 0.4, colors.HexColor("#cccccc")),
        ("VALIGN",        (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING",    (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING",   (0, 0), (-1, -1), 5),
    ]
    for idx, bg in enumerate(row_bgs[1:], 1):
        style_cmds.append(("BACKGROUND", (0, idx), (-1, idx), bg))

    field_tbl.setStyle(TableStyle(style_cmds))
    story.append(field_tbl)
    story.append(Spacer(1, 10*mm))
    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.grey))
    story.append(Spacer(1, 5*mm))

    #  Signature row
    sig_tbl = Table(
        [
            [Paragraph("Applicant Signature", lbl_s),
             Paragraph("Date", lbl_s)],
            [Paragraph("_________________________", val_s),
             Paragraph(date.today().strftime("%d / %m / %Y"), val_s)],
        ],
        colWidths=[85*mm, 85*mm]
    )
    sig_tbl.setStyle(TableStyle([
        ("VALIGN",      (0, 0), (-1, -1), "BOTTOM"),
        ("TOPPADDING",  (0, 0), (-1, -1), 3),
    ]))
    story.append(sig_tbl)

    doc.build(story)
    print(f"  ReportLab PDF generated -> {output_path}")

# Public API
def fill_and_save_pdf(pdf_path, user_data, scheme_name,
                      auto_filled=None, profile=None):
    """
    Write the filled form to disk.

    Tries AcroForm fill first; falls back to ReportLab generation.
    Output filename: filled_<scheme_name>.pdf  in the current directory.

    Parameters
    ----------
    pdf_path    : path to the original (empty) form PDF
    user_data   : {field_name: value} from collect_user_inputs()
    scheme_name : chosen scheme name (used in filename and PDF title)
    auto_filled : {field_name: value_or_None} from autofill_fields()
    profile     : user profile dict (used only for colour coding in ReportLab)
    """
    if auto_filled is None:
        auto_filled = {}
    if profile is None:
        profile = {}

    # Build a safe filename (replace spaces and slashes)
    safe_name   = (scheme_name.lower()
                   .replace(" ", "_")
                   .replace("/", "_")
                   .replace("\\", "_"))
    output_path = f"filled_{safe_name}.pdf"

    print()
    print("=" * 60)
    print("  GENERATING FILLED PDF")
    print("=" * 60)

    #  AcroForm fill ---
    if _is_fillable(pdf_path):
        print("  Detected fillable AcroForm PDF.")
        print("  Writing field values directly into the form ...")
        try:
            _fill_acroform(pdf_path, user_data, output_path)
            _print_success(output_path)
            return
        except Exception as exc:
            print(f"  [warning] AcroForm fill failed: {exc}")
            print("  Falling back to ReportLab PDF generation ...")

    #  ReportLab new PDF 
    print("  Generating a new structured PDF with ReportLab ...")
    try:
        _generate_reportlab_pdf(
            user_data, scheme_name, auto_filled, profile, output_path
        )
        _print_success(output_path)
    except Exception as exc:
        print(f"  [ERROR] PDF generation failed: {exc}")
        print("  Check that reportlab is installed:  pip install reportlab")


def _print_success(output_path):
    abs_path = os.path.abspath(output_path)
    print()
    print("=" * 60)
    print("  SUCCESS")
    print(f"  Filled form saved as:")
    print(f"  {abs_path}")
    print("=" * 60)
