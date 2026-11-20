"""Main entry point for the Automated Resume Screening Tool."""

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


def create_sample_resumes_if_empty(resumes_dir: Path) -> None:
    """Create sample TXT resumes if the resumes folder has no supported files."""
    resumes_dir.mkdir(parents=True, exist_ok=True)
    existing = [p for p in resumes_dir.iterdir() if p.suffix.lower() in {".pdf", ".docx", ".txt"}]
    if existing:
        return

    samples = {
        "candidate_01_ananya_sharma.txt": (
            "Ananya Sharma\n"
            "2 years of experience as a Data Analyst.\n"
            "Skills: Python, SQL, Pandas, NumPy, Excel, Power BI, Git, Statistics.\n"
            "Built automated ETL scripts in Python and Pandas and interactive Power BI dashboards."
        ),
        "candidate_02_rohan_verma.txt": (
            "Rohan Verma\n"
            "1.5 years of experience in reporting and analytics.\n"
            "Skills: Python, SQL, Excel, Tableau, Data Visualization.\n"
            "Automated weekly Excel and SQL reports for sales operations."
        ),
        "candidate_03_priya_nair.txt": (
            "Priya Nair\n"
            "1 year of experience working with data.\n"
            "Skills: Python, Pandas, Scikit-Learn, Machine Learning, Git.\n"
            "Developed predictive models in Python and Pandas."
        ),
        "candidate_04_kabir_singh.txt": (
            "Kabir Singh\n"
            "Fresher / 0.5 years internship experience.\n"
            "Skills: HTML/CSS, Java, C++.\n"
            "Built basic web pages and academic assignments."
        ),
    }
    for filename, content in samples.items():
        (resumes_dir / filename).write_text(content, encoding="utf-8")


def load_job_description(jd_path: Path) -> dict:
    """Load job description JSON configuration."""
    if not jd_path.exists():
        raise FileNotFoundError(f"Job description not found at: {jd_path}")
    return json.loads(jd_path.read_text(encoding="utf-8"))


def run_pipeline() -> pd.DataFrame:
    """Execute end-to-end resume screening and save CSV report."""
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    create_sample_resumes_if_empty(RESUMES_DIR)
    jd = load_job_description(JD_PATH)

    must_have = jd.get("must_have_skills", [])
    nice_to_have = jd.get("nice_to_have_skills", [])
    min_exp = float(jd.get("min_exp_years", 1.0))
    shortlist_thresh = float(jd.get("shortlist_threshold", 65.0))
    consider_thresh = float(jd.get("consider_threshold", 45.0))
    cleaned_jd = clean_text(jd.get("description", ""))

    resume_files = sorted(
        [p for p in RESUMES_DIR.iterdir() if p.is_file() and p.suffix.lower() in {".pdf", ".docx", ".txt"}]
    )

    if not resume_files:
        print(f"No resume files found in {RESUMES_DIR}")
        return pd.DataFrame()

    records = []
    cleaned_texts = []

    for file_path in resume_files:
        raw_text, err = extract_resume_text(file_path)
        cleaned = clean_text(raw_text) if not err else ""
        cleaned_texts.append(cleaned)
        records.append({
            "resume_file": file_path.name,
            "candidate_name": file_path.stem.replace("_", " ").title(),
            "extraction_error": err,
            "cleaned_text": cleaned,
        })

    similarities = compute_tfidf_similarities(cleaned_jd, cleaned_texts)

    final_rows = []
    for rec, sim in zip(records, similarities):
        if rec["extraction_error"]:
            final_rows.append({
                "rank": 0,
                "candidate_name": rec["candidate_name"],
                "resume_file": rec["resume_file"],
                "match_score": 0.0,
                "status": "ERROR",
                "must_have_coverage": f"0/{len(must_have)}",
                "missing_must_have": ", ".join(must_have),
                "matched_skills": "",
                "years_experience": 0.0,
                "tfidf_similarity": 0.0,
                "notes": rec["extraction_error"],
            })
            continue

        skill_info = evaluate_candidate_skills(
            rec["cleaned_text"],
            must_have_skills=must_have,
            nice_to_have_skills=nice_to_have,
        )
        score_info = score_candidate(
            tfidf_sim=sim,
            matched_must=skill_info["matched_must_have"],
            total_must=len(must_have),
            matched_nice=skill_info["matched_nice_to_have"],
            total_nice=len(nice_to_have),
            years_exp=skill_info["years_experience"],
            min_exp_years=min_exp,
            shortlist_threshold=shortlist_thresh,
            consider_threshold=consider_thresh,
        )

        missing_list = skill_info["missing_must_have"]
        notes = (
            f"Missing must-have: {', '.join(missing_list)}"
            if missing_list
            else "All must-have skills matched"
        )

        final_rows.append({
            "candidate_name": rec["candidate_name"],
            "resume_file": rec["resume_file"],
            "match_score": score_info["match_score"],
            "status": score_info["status"],
            "must_have_coverage": score_info["must_have_coverage"],
            "missing_must_have": ", ".join(missing_list) if missing_list else "None",
            "matched_skills": ", ".join(skill_info["matched_skills"]),
            "years_experience": skill_info["years_experience"],
            "tfidf_similarity": score_info["tfidf_similarity"],
            "notes": notes,
        })

    df = pd.DataFrame(final_rows)
    df = df.sort_values(by=["match_score", "years_experience"], ascending=[False, False]).reset_index(drop=True)
    df.insert(0, "rank", df.index + 1)

    df.to_csv(REPORT_PATH, index=False)

    print("\n" + "=" * 72)
    print(f"JOB ROLE: {jd.get('job_title', 'Role')}")
    print("=" * 72)
    display_cols = ["rank", "candidate_name", "match_score", "status", "must_have_coverage", "missing_must_have"]
    print(df[display_cols].to_string(index=False))
    print("-" * 72)
    print(f"CSV report saved to: {REPORT_PATH}\n")
    return df


if __name__ == "__main__":
    run_pipeline()