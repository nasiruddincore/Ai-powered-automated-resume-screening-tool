"""Optional Streamlit UI for the Automated Resume Screening Tool."""

import json
from pathlib import Path
import pandas as pd
import streamlit as st

from src.extractor import extract_resume_text
from src.cleaner import clean_text
from src.matcher import evaluate_candidate_skills
from src.scorer import compute_tfidf_similarities, score_candidate

st.set_page_config(page_title="Automated Resume Screening Tool", layout="wide")
st.title("Automated Resume Screening Tool")
st.caption("Upload sample resumes (PDF, DOCX, TXT) and screen them against a job description.")

jd_path = Path("data/job_description.json")
default_jd = json.loads(jd_path.read_text(encoding="utf-8")) if jd_path.exists() else {}

with st.sidebar:
    st.header("Job Description Settings")
    role_title = st.text_input("Job Title", value=default_jd.get("job_title", "Data Analyst"))
    min_exp = st.number_input("Minimum Experience (Years)", min_value=0.0, max_value=20.0, value=float(default_jd.get("min_exp_years", 1.0)), step=0.5)
    must_have_raw = st.text_input("Must-Have Skills (comma-separated)", value=", ".join(default_jd.get("must_have_skills", ["python", "sql", "pandas", "excel"])))
    nice_have_raw = st.text_input("Nice-to-Have Skills (comma-separated)", value=", ".join(default_jd.get("nice_to_have_skills", ["power bi", "tableau", "git"])))
    jd_desc = st.text_area("Job Description Text", value=default_jd.get("description", ""), height=160)

uploaded_files = st.file_uploader("Upload Resumes", type=["pdf", "docx", "txt"], accept_multiple_files=True)

if st.button("Run Screening", type="primary"):
    if not uploaded_files:
        st.warning("Please upload at least one PDF, DOCX, or TXT resume.")
    else:
        must_have = [s.strip().lower() for s in must_have_raw.split(",") if s.strip()]
        nice_have = [s.strip().lower() for s in nice_have_raw.split(",") if s.strip()]
        cleaned_jd = clean_text(jd_desc)

        temp_dir = Path("outputs/_streamlit_uploads")
        temp_dir.mkdir(parents=True, exist_ok=True)

        parsed_items = []
        cleaned_resumes = []

        for up_file in uploaded_files:
            temp_path = temp_dir / up_file.name
            temp_path.write_bytes(up_file.getbuffer())
            raw_txt, err = extract_resume_text(temp_path)
            cleaned_txt = clean_text(raw_txt) if not err else ""
            cleaned_resumes.append(cleaned_txt)
            parsed_items.append((up_file.name, cleaned_txt, err))

        sims = compute_tfidf_similarities(cleaned_jd, cleaned_resumes)
        rows = []

        for (fname, c_txt, err), sim in zip(parsed_items, sims):
            if err:
                rows.append({"Candidate File": fname, "Match Score": 0.0, "Status": "ERROR", "Notes": err})
                continue
            skill_res = evaluate_candidate_skills(c_txt, must_have, nice_have)
            score_res = score_candidate(
                tfidf_sim=sim,
                matched_must=skill_res["matched_must_have"],
                total_must=len(must_have),
                matched_nice=skill_res["matched_nice_to_have"],
                total_nice=len(nice_have),
                years_exp=skill_res["years_experience"],
                min_exp_years=min_exp,
            )
            rows.append({
                "Candidate File": fname,
                "Match Score": score_res["match_score"],
                "Status": score_res["status"],
                "Must-Have Coverage": score_res["must_have_coverage"],
                "Missing Must-Have": ", ".join(skill_res["missing_must_have"]) or "None",
                "Matched Skills": ", ".join(skill_res["matched_skills"]),
                "Exp (Yrs)": skill_res["years_experience"],
                "TF-IDF Sim": score_res["tfidf_similarity"],
            })

        result_df = pd.DataFrame(rows).sort_values(by="Match Score", ascending=False).reset_index(drop=True)
        result_df.insert(0, "Rank", result_df.index + 1)

        st.subheader(f"Results for: {role_title}")
        st.dataframe(result_df, use_container_width=True)

        csv_bytes = result_df.to_csv(index=False).encode("utf-8")
        st.download_button("Download CSV Report", data=csv_bytes, file_name="screening_report.csv", mime="text/csv")