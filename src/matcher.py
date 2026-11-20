"""Extract skills and estimate years of experience from cleaned resume text."""

import re
from typing import Dict, List, Set

# Curated skill catalog with aliases normalized to canonical skill names
SKILL_ALIASES: Dict[str, List[str]] = {
    "python": ["python", "py"],
    "sql": ["sql", "mysql", "postgresql", "postgres", "sqlite", "t-sql"],
    "pandas": ["pandas"],
    "numpy": ["numpy"],
    "excel": ["excel", "ms excel", "microsoft excel", "spreadsheets"],
    "power bi": ["power bi", "powerbi"],
    "tableau": ["tableau"],
    "scikit-learn": ["scikit-learn", "sklearn", "scikit learn"],
    "machine learning": ["machine learning", "ml"],
    "data visualization": ["data visualization", "visualization", "charts", "dashboards"],
    "statistics": ["statistics", "statistical analysis", "a/b testing", "hypothesis testing"],
    "git": ["git", "github"],
    "java": ["java"],
    "c++": ["c++", "cpp"],
    "html/css": ["html", "css"],
    "fastapi": ["fastapi"],
    "flask": ["flask"],
    "django": ["django"],
}


def extract_skills(cleaned_text: str, extra_skills: List[str] = None) -> List[str]:
    """Return sorted list of matched canonical skills from cleaned text."""
    found: Set[str] = set()
    padded = f" {cleaned_text} "

    for canonical, aliases in SKILL_ALIASES.items():
        for alias in aliases:
            escaped = re.escape(alias.lower())
            # Match whole phrase/token safely
            pattern = rf"(?<![a-z0-9\+\#]){escaped}(?![a-z0-9\+\#])"
            if re.search(pattern, padded):
                found.add(canonical)
                break

    if extra_skills:
        for skill in extra_skills:
            s_clean = skill.strip().lower()
            if not s_clean:
                continue
            escaped = re.escape(s_clean)
            pattern = rf"(?<![a-z0-9\+\#]){escaped}(?![a-z0-9\+\#])"
            if re.search(pattern, padded):
                found.add(s_clean)

    return sorted(found)


def estimate_experience_years(cleaned_text: str) -> float:
    """Estimate years of experience mentioned in resume text."""
    matches = []
    patterns = [
        r"(\d+(?:\.\d+)?)\s*\+?\s*(?:years|yrs|year|yr)\s+(?:of\s+)?(?:experience|exp)?",
        r"(?:experience|exp)\s*[:\-]?\s*(\d+(?:\.\d+)?)\s*\+?\s*(?:years|yrs|year|yr)",
    ]
    for pat in patterns:
        for m in re.finditer(pat, cleaned_text):
            try:
                val = float(m.group(1))
                # Ignore unrealistic numbers like graduation year 2026
                if 0.0 <= val <= 40.0:
                    matches.append(val)
            except ValueError:
                continue

    if matches:
        return round(max(matches), 1)

    # Fallback check for fresher / internship mentions
    if "intern" in cleaned_text or "fresher" in cleaned_text:
        return 0.5
    return 0.0


def evaluate_candidate_skills(
    cleaned_text: str,
    must_have_skills: List[str],
    nice_to_have_skills: List[str],
) -> Dict[str, object]:
    """Compare extracted resume skills against must-have and nice-to-have lists."""
    must_norm = [s.strip().lower() for s in must_have_skills if s.strip()]
    nice_norm = [s.strip().lower() for s in nice_to_have_skills if s.strip()]

    matched_all = extract_skills(cleaned_text, extra_skills=must_norm + nice_norm)
    matched_set = set(matched_all)

    matched_must = [s for s in must_norm if s in matched_set]
    missing_must = [s for s in must_norm if s not in matched_set]
    matched_nice = [s for s in nice_norm if s in matched_set]
    exp_years = estimate_experience_years(cleaned_text)

    return {
        "matched_skills": matched_all,
        "matched_must_have": matched_must,
        "missing_must_have": missing_must,
        "matched_nice_to_have": matched_nice,
        "years_experience": exp_years,
    }