"""
scheme_search.py
================
Searches DuckDuckGo for Indian pension schemes relevant to the user's profile,
then uses Ollama (llama3:8b) to reason over the results and recommend the top 3.

No API key required.  DuckDuckGo search is free and anonymous.

Exported
--------
  get_pension_recommendations(profile) -> list[dict]
"""

import json
import re

# DuckDuckGo search - try both import paths used across different versions
try:
    from langchain_community.tools import DuckDuckGoSearchRun
except ImportError:
    try:
        from langchain.tools import DuckDuckGoSearchRun
    except ImportError:
        raise ImportError(
            "DuckDuckGoSearchRun not found.\n"
            "Install with:  pip install duckduckgo-search langchain-community"
        )

from langchain_ollama import OllamaLLM

# Module-level singletons  (created once, reused across calls)
def _make_llm():
    try:
        return OllamaLLM(model="llama3:8b", temperature=0.2)
    except Exception as exc:
        raise RuntimeError(
            f"Could not connect to Ollama: {exc}\n"
            "Make sure Ollama is running:  ollama serve\n"
            "And the model is available:   ollama pull llama3:8b"
        )


llm         = _make_llm()
search_tool = DuckDuckGoSearchRun()

# Search helpers
def _build_queries(profile):
    """
    Return three search queries.  The first is specific to the user profile;
    the others broaden the net to catch common schemes.
    """
    return [
        (
            f"Indian pension scheme {profile['occupation']} "
            f"age {profile['age']} income {profile['income_lpa']} LPA "
            f"{profile['state']} state eligibility 2024"
        ),
        (
            f"best government pension India {profile['occupation']} "
            "eligibility documents how to apply"
        ),
        "Atal Pension Yojana NPS SCSS EPF PM-SYM India comparison eligibility 2024",
    ]


def _run_searches(queries):
    """
    Execute each query with DuckDuckGoSearchRun.
    Silently skip any query that raises an exception.
    Returns a single concatenated string of all snippets.
    """
    snippets = []
    for q in queries:
        try:
            result = search_tool.run(q)
            if result:
                snippets.append(result)
        except Exception as exc:
            print(f"  [search warning] Query failed: {exc}")
    return "\n\n---\n\n".join(snippets)

# LLM output parsing

def _parse_json_schemes(raw):
    """
    Try to extract a JSON array from the LLM's raw text output.

    Two attempts:
      1.  Find the outermost [...] block with a greedy regex and json.loads it.
          (Greedy is intentional: the array may span many lines.)
      2.  Line-by-line fallback that reconstructs partial dicts from
          labelled text lines (handles prose-style LLM responses).

    Returns a list of dicts (may be empty if nothing could be parsed).
    """
    # --- Attempt 1: JSON array ---
    match = re.search(r"\[.*\]", raw, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group())
            if isinstance(data, list) and data:
                # Normalise keys
                cleaned = []
                for item in data:
                    if isinstance(item, dict):
                        cleaned.append({
                            "name"               : item.get("name", ""),
                            "eligibility"        : item.get("eligibility", "N/A"),
                            "pros"               : item.get("pros", "N/A"),
                            "cons"               : item.get("cons", "N/A"),
                            "documents"          : item.get("documents", "N/A"),
                            "where_to_apply"     : item.get("where_to_apply", "N/A"),
                            "application_process": item.get("application_process", "N/A"),
                        })
                if cleaned:
                    return cleaned
        except (json.JSONDecodeError, ValueError):
            pass

    # --- Attempt 2: line-by-line reconstruction ---
    schemes = []
    blocks  = re.split(r"\n(?=\d+\.|Option\s*\d|Scheme\s*\d)", raw)

    for block in blocks[:3]:
        name_m = re.search(r"(?:name|scheme)\s*[:\-]\s*([^\n]+)", block, re.I)
        if not name_m:
            continue
        schemes.append({
            "name"               : name_m.group(1).strip(),
            "eligibility"        : _grab(block, r"eligibility"),
            "pros"               : _grab(block, r"pros?|advantages?|benefits?"),
            "cons"               : _grab(block, r"cons?|disadvantages?|limitations?"),
            "documents"          : _grab(block, r"documents?|papers?|required"),
            "where_to_apply"     : _grab(block, r"where|apply\s*at|office|bank|portal"),
            "application_process": _grab(block, r"process|how\s*to|steps?"),
        })

    return schemes


def _grab(text, pattern):
    """Extract the first non-empty value after a label matching pattern."""
    m = re.search(rf"(?:{pattern})\s*[:\-]\s*([^\n]+)", text, re.I)
    return m.group(1).strip() if m else "N/A"

# Public API
def get_pension_recommendations(profile):
    """
    Search the web for pension schemes and use the LLM to recommend the best 3
    for the given user profile.

    Parameters
    ----------
    profile : dict  with keys name, age, gender, occupation, income_lpa, state

    Returns
    -------
    list of up to 3 dicts, each with keys:
        name, eligibility, pros, cons, documents, where_to_apply,
        application_process
    Returns [] on any unrecoverable error.
    """
    # gather live search data
    queries = _build_queries(profile)
    print("  Running DuckDuckGo searches ...")
    search_text = _run_searches(queries)

    if not search_text.strip():
        print("  [warning] All searches returned empty results.")
        search_text = "No search results available. Use general knowledge."

    # ask the LLM to analyse and recommend
    prompt = (
        "You are an expert advisor on Indian government pension and savings schemes.\n\n"
        "USER PROFILE:\n"
        f"  Name        : {profile['name']}\n"
        f"  Age         : {profile['age']}\n"
        f"  Gender      : {profile['gender']}\n"
        f"  Occupation  : {profile['occupation']}\n"
        f"  Annual Income: {profile['income_lpa']} LPA\n"
        f"  State       : {profile['state']}\n\n"
        "SEARCH RESULTS (live web data):\n"
        f"{search_text[:3000]}\n\n"
        "TASK:\n"
        "Based on the profile and search results, identify the TOP 3 most suitable\n"
        "Indian pension/savings schemes for this person.\n\n"
        "Return ONLY a valid JSON array.  No explanation, no markdown, no preamble.\n"
        "Each object must have exactly these 7 keys:\n"
        "  name, eligibility, pros, cons, documents, where_to_apply, application_process\n\n"
        "Example format:\n"
        '[\n'
        '  {\n'
        '    "name": "Atal Pension Yojana",\n'
        '    "eligibility": "Age 18-40, Indian citizen, savings bank account",\n'
        '    "pros": "Guaranteed pension, government co-contribution",\n'
        '    "cons": "Low pension amount (max Rs 5000/month)",\n'
        '    "documents": "Aadhaar, mobile number linked to bank account",\n'
        '    "where_to_apply": "Any nationalised bank or post office",\n'
        '    "application_process": "Visit bank with Aadhaar, fill APY form, link savings account"\n'
        '  }\n'
        ']\n\n'
        "Output the JSON array now:"
    )

    print("  Asking Ollama (llama3:8b) to analyse and recommend ...")
    try:
        raw_output = llm.invoke(prompt)
    except Exception as exc:
        print(f"  [LLM error] {exc}")
        return []

    schemes = _parse_json_schemes(raw_output)

    if not schemes:
        print("  [warning] Could not parse structured output from LLM.")
        print("  Raw LLM output (first 800 chars):")
        print("  " + raw_output[:800].replace("\n", "\n  "))

    return schemes[:3]
