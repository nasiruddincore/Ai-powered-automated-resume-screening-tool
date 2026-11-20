import re
from typing import Dict, List, Set

SKILL_ALIASES: Dict[str, List[str]] = {
    "python": ["python", "py"],
    "sql": ["sql", "mysql", "postgresql", "postgres", "sqlite"],
    "pandas": ["pandas"],
    "numpy": ["numpy"],
    "excel": ["excel", "ms excel", "microsoft excel", "spreadsheets"],
    "power bi": ["power bi", "powerbi"],
    "tableau": ["tableau"],
    "scikit-learn": ["scikit-learn", "sklearn", "scikit learn"],
    "machine learning": ["machine learning", "ml"],
    "data visualization": ["data visualization", "visualization", "dashboards"],
    "statistics": ["statistics", "statistical analysis"],
    "git": ["git", "github"],
    "java": ["java"],
    "c++": ["c++", "cpp"],
    "html/css": ["html", "css"],
}


def extract_skills(cleaned_text: str, extra_skills: List[str] = None) -> List[str]:
    found: Set[str] = set()
    padded = f" {cleaned_text} "
    for canonical, aliases in SKILL_ALIASES.items():
        for alias in aliases:
            pat = rf"(?<![a-z0-9\+\#]){re.escape(alias.lower())}(?![a-z0-9\+\#])"
            if re.search(pat, padded):
                found.add(canonical)
                break
    for skill in (extra_skills or []):
        s = skill.strip().lower()
        if s:
            pat = rf"(?<![a-z0-9\+\#]){re.escape(s)}(?![a-z0-9\+\#])"
            if re.search(pat, padded):
                found.add(s)
    return sorted(found)


def estimate_experience_years(cleaned_text: str) -> float:
    vals = []
    for m in re.finditer(r"(\d+(?:\.\d+)?)\s*\+?\s*(?:years|yrs|year|yr)", cleaned_text):
        v = float(m.group(1))
        if 0.0 <= v <= 40.0:
            vals.append(v)
    if vals:
        return round(max(vals), 1)
    if "intern" in cleaned_text or "fresher" in cleaned_text:
        return 0.5
    return 0.0


def evaluate_candidate_skills(cleaned_text: str, must_have_skills: List[str], nice_to_have_skills: List[str]) -> Dict[str, object]:
    must_norm = [s.strip().lower() for s in must_have_skills if s.strip()]
    nice_norm = [s.strip().lower() for s in nice_to_have_skills if s.strip()]
    matched_all = extract_skills(cleaned_text, extra_skills=must_norm + nice_norm)
    mset = set(matched_all)
    return {
        "matched_skills": matched_all,
        "matched_must_have": [s for s in must_norm if s in mset],
        "missing_must_have": [s for s in must_norm if s not in mset],
        "matched_nice_to_have": [s for s in nice_norm if s in mset],
        "years_experience": estimate_experience_years(cleaned_text),
    }
