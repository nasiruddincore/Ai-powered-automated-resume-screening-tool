import json
from pathlib import Path
import pandas as pd

from src.extractor import extract_resume_text
from src.cleaner import clean_text
from src.matcher import evaluate_candidate_skills
from src.scorer import compute_tfidf_similarities, score_candidate

BASE_DIR = Path(__file__).resolve().parent
RESUMES_DIR = BASE_DIR / "resumes"
JD_PATH = BASE_DIR / "data" / "job_description.json"
OUTPUTS_DIR = BASE_DIR / "outputs"
REPORT_PATH = OUTPUTS_DIR / "screening_report.csv"


def seed_resumes() -> None:
    RESUMES_DIR.mkdir(parents=True, exist_ok=True)
    if any(p.suffix.lower() in {".pdf", ".docx", ".txt"} for p in RESUMES_DIR.iterdir()):
        return
    samples = {
        "ananya_sharma.txt": "Ananya Sharma. 2 years of experience. Skills: Python, SQL, Pandas, Excel, Power BI, Git.",
        "rohan_verma.txt": "Rohan Verma. 1.5 years of experience. Skills: Python, SQL, Excel, Tableau.",
        "kabir_singh.txt": "Kabir Singh. Fresher. Skills: HTML, CSS, Java, C++.",
    }
    for name, txt in samples.items():
        (RESUMES_DIR / name).write_text(txt, encoding="utf-8")


def main() -> None:
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    seed_resumes()
    jd = json.loads(JD_PATH.read_text(encoding="utf-8"))
    must_have = jd.get("must_have_skills", [])
    nice_have = jd.get("nice_to_have_skills", [])
    min_exp = float(jd.get("min_exp_years", 1.0))
    cleaned_jd = clean_text(jd.get("description", ""))

    files = sorted([p for p in RESUMES_DIR.iterdir() if p.suffix.lower() in {".pdf", ".docx", ".txt"}])
    parsed, cleaned_list = [], []
    for f in files:
        raw, err = extract_resume_text(f)
        c = clean_text(raw) if not err else ""
        cleaned_list.append(c)
        parsed.append((f.name, f.stem.replace("_", " ").title(), c, err))

    sims = compute_tfidf_similarities(cleaned_jd, cleaned_list)
    rows = []
    for (fname, cname, c_txt, err), sim in zip(parsed, sims):
        if err:
            rows.append({"candidate_name": cname, "resume_file": fname, "match_score": 0.0, "status": "ERROR", "missing_must_have": err})
            continue
        sk = evaluate_candidate_skills(c_txt, must_have, nice_have)
        sc = score_candidate(sim, sk["matched_must_have"], len(must_have), sk["matched_nice_to_have"], len(nice_have), sk["years_experience"], min_exp)
        rows.append({
            "candidate_name": cname,
            "resume_file": fname,
            "match_score": sc["match_score"],
            "status": sc["status"],
            "must_have_coverage": sc["must_have_coverage"],
            "missing_must_have": ", ".join(sk["missing_must_have"]) or "None",
            "matched_skills": ", ".join(sk["matched_skills"]),
            "years_experience": sk["years_experience"],
            "tfidf_similarity": sc["tfidf_similarity"],
        })

    df = pd.DataFrame(rows).sort_values(by="match_score", ascending=False).reset_index(drop=True)
    df.insert(0, "rank", df.index + 1)
    df.to_csv(REPORT_PATH, index=False)
    print(df[["rank", "candidate_name", "match_score", "status", "must_have_coverage", "missing_must_have"]].to_string(index=False))
    print(f"\nSaved report to: {REPORT_PATH}")


if __name__ == "__main__":
    main()
