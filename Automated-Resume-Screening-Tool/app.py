import json
from pathlib import Path
import pandas as pd
import streamlit as st

from src.extractor import extract_resume_text
from src.cleaner import clean_text
from src.matcher import evaluate_candidate_skills
from src.scorer import compute_tfidf_similarities, score_candidate

st.set_page_config(
    page_title="Automated Resume Screening Tool",
    page_icon="🎯",
    layout="wide",
)

st.markdown(
    """
    <style>
        /* Multi-color background & crisp high-contrast typography */
        .stApp {
            background: linear-gradient(135deg, #090d1e 0%, #1b1446 50%, #2d1042 100%);
            color: #f8fafc !important;
        }

        label, .stMarkdown, p, span, div {
            color: #f8fafc;
        }

        /* High-contrast input fields & textarea */
        .stTextInput input, .stTextArea textarea, .stNumberInput input {
            background-color: #111827 !important;
            color: #ffffff !important;
            border: 1.5px solid #38bdf8 !important;
            border-radius: 10px !important;
            font-size: 0.98rem !important;
            font-weight: 500 !important;
        }

        .stTextInput input:focus, .stTextArea textarea:focus, .stNumberInput input:focus {
            border-color: #f472b6 !important;
            box-shadow: 0 0 0 2px rgba(244, 114, 182, 0.35) !important;
        }

        /* File uploader high-contrast styling */
        [data-testid="stFileUploader"] section {
            background: linear-gradient(135deg, rgba(30, 41, 59, 0.96), rgba(49, 46, 129, 0.88)) !important;
            border: 2px dashed #38bdf8 !important;
            border-radius: 14px !important;
            color: #ffffff !important;
        }
        [data-testid="stFileUploader"] section span,
        [data-testid="stFileUploader"] section small,
        [data-testid="stFileUploader"] section div {
            color: #f1f5f9 !important;
        }
        [data-testid="stFileUploader"] button {
            background: linear-gradient(90deg, #2563eb, #7c3aed) !important;
            color: #ffffff !important;
            border: none !important;
            font-weight: 700 !important;
        }

        /* Primary button */
        .stButton > button[kind="primary"] {
            background: linear-gradient(90deg, #ec4899 0%, #8b5cf6 50%, #06b6d4 100%) !important;
            color: #ffffff !important;
            font-weight: 800 !important;
            border: none !important;
            border-radius: 10px !important;
            padding: 0.55rem 1.4rem !important;
        }

        /* Hero Header */
        .hero-box {
            background: linear-gradient(90deg, #2563eb 0%, #9333ea 52%, #ec4899 100%);
            padding: 1.4rem 1.8rem;
            border-radius: 16px;
            margin-bottom: 1.15rem;
            box-shadow: 0 10px 26px rgba(147, 51, 234, 0.35);
            border: 1px solid rgba(255, 255, 255, 0.25);
        }
        .hero-box h1 {
            margin: 0;
            font-size: 1.85rem;
            font-weight: 800;
            color: #ffffff !important;
        }
        .hero-box p {
            margin: 0.35rem 0 0 0;
            color: #fef9c3 !important;
            font-size: 0.98rem;
            font-weight: 500;
        }

        /* Section Cards */
        .color-card {
            background: rgba(15, 23, 42, 0.9);
            border: 1px solid rgba(129, 140, 248, 0.35);
            border-radius: 14px;
            padding: 1rem 1.25rem;
            margin-bottom: 1rem;
        }
        .sec-must {
            color: #38bdf8 !important;
            font-weight: 800;
            font-size: 1.02rem;
            margin-bottom: 0.45rem;
        }
        .sec-nice {
            color: #f472b6 !important;
            font-weight: 800;
            font-size: 1.02rem;
            margin-bottom: 0.45rem;
        }
        .sec-jd {
            color: #fbbf24 !important;
            font-weight: 800;
            font-size: 1.02rem;
            margin-top: 0.4rem;
            margin-bottom: 0.45rem;
        }

        /* Colorful Badges */
        .badge {
            display: inline-block;
            padding: 0.24rem 0.72rem;
            border-radius: 999px;
            font-size: 0.8rem;
            font-weight: 700;
            margin-right: 0.38rem;
            margin-top: 0.4rem;
            color: #ffffff !important;
        }
        .b-must { background: linear-gradient(135deg, #0284c7, #2563eb); }
        .b-nice { background: linear-gradient(135deg, #9333ea, #db2777); }
        .b-shortlist { background: linear-gradient(135deg, #059669, #10b981); }
        .b-consider { background: linear-gradient(135deg, #d97706, #f59e0b); }
        .b-reject { background: linear-gradient(135deg, #e11d48, #be123c); }
        .b-ok { background: linear-gradient(135deg, #0d9488, #06b6d4); }
        .b-gap { background: linear-gradient(135deg, #dc2626, #f97316); }

        /* KPI & Candidate Cards */
        .kpi-card {
            border-radius: 14px;
            padding: 1rem 1.2rem;
            color: #ffffff !important;
            margin-bottom: 0.75rem;
            box-shadow: 0 8px 20px rgba(0, 0, 0, 0.3);
        }
        .kpi-cyan { background: linear-gradient(135deg, #0284c7, #2563eb); }
        .kpi-emerald { background: linear-gradient(135deg, #059669, #10b981); }
        .kpi-amber { background: linear-gradient(135deg, #d97706, #f59e0b); }
        .kpi-pink { background: linear-gradient(135deg, #c026d3, #ec4899); }
        .kpi-label { font-size: 0.8rem; font-weight: 700; text-transform: uppercase; color: #ffffff !important; }
        .kpi-val { font-size: 1.75rem; font-weight: 800; margin-top: 0.2rem; color: #ffffff !important; }

        .cand-card {
            background: rgba(15, 23, 42, 0.92);
            border: 1px solid rgba(148, 163, 184, 0.25);
            border-left: 6px solid #38bdf8;
            border-radius: 12px;
            padding: 1rem 1.25rem;
            margin-bottom: 0.85rem;
        }
        .cand-shortlist { border-left-color: #10b981; }
        .cand-consider { border-left-color: #f59e0b; }
        .cand-reject { border-left-color: #f43f5e; }
    </style>
    """,
    unsafe_allow_html=True,
)

DEFAULT_MUST_HAVE = ["python", "sql", "pandas", "excel"]
DEFAULT_NICE_TO_HAVE = ["numpy", "power bi", "tableau", "scikit-learn", "git", "data visualization", "statistics"]
DEFAULT_JD = (
    "Looking for a detail-oriented candidate with strong Python, SQL, Pandas, and Excel skills "
    "to clean data, automate recurring reports, build dashboards, and support analytics workflows. "
    "Experience with Power BI, Tableau, NumPy, scikit-learn, and Git is a plus."
)

jd_file = Path("data/job_description.json")
jd_defaults = json.loads(jd_file.read_text(encoding="utf-8-sig")) if jd_file.exists() else {}

st.markdown(
    """
    <div class="hero-box">
        <h1>🎯 Automated Resume Screening Command Center</h1>
        <p>Multi-color NLP &amp; TF-IDF candidate ranker with must-have skill gap detection and instant CSV reporting.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown("## 🎛️ Scoring Controls")
    role_title = st.text_input("Job Title", value=jd_defaults.get("job_title", "Junior Data Analyst / Python Automation"))
    min_exp = st.number_input(
        "Minimum Experience (Years)",
        min_value=0.0,
        max_value=25.0,
        value=float(jd_defaults.get("min_exp_years", 1.0)),
        step=0.5,
    )
    shortlist_th = st.slider(
        "Shortlist Threshold",
        min_value=30.0,
        max_value=95.0,
        value=float(jd_defaults.get("shortlist_threshold", 65.0)),
        step=1.0,
    )
    consider_th = st.slider(
        "Consider Threshold",
        min_value=15.0,
        max_value=80.0,
        value=float(jd_defaults.get("consider_threshold", 45.0)),
        step=1.0,
    )

st.markdown('<div class="color-card">', unsafe_allow_html=True)
c1, c2 = st.columns(2)
with c1:
    st.markdown('<div class="sec-must">🔥 Must-Have Skills</div>', unsafe_allow_html=True)
    must_raw = st.text_input(
        "Must-Have Skills",
        value=", ".join(jd_defaults.get("must_have_skills", DEFAULT_MUST_HAVE)),
        label_visibility="collapsed",
    )
    must_chips = [s.strip() for s in must_raw.split(",") if s.strip()]
    st.markdown("".join(f'<span class="badge b-must">{s}</span>' for s in must_chips), unsafe_allow_html=True)

with c2:
    st.markdown('<div class="sec-nice">✨ Nice-to-Have Skills</div>', unsafe_allow_html=True)
    nice_raw = st.text_input(
        "Nice-to-Have Skills",
        value=", ".join(jd_defaults.get("nice_to_have_skills", DEFAULT_NICE_TO_HAVE)),
        label_visibility="collapsed",
    )
    nice_chips = [s.strip() for s in nice_raw.split(",") if s.strip()]
    st.markdown("".join(f'<span class="badge b-nice">{s}</span>' for s in nice_chips), unsafe_allow_html=True)

st.markdown('<div class="sec-jd">📝 Job Description</div>', unsafe_allow_html=True)
jd_desc = st.text_area(
    "Job Description",
    value=jd_defaults.get("description", DEFAULT_JD),
    height=120,
    label_visibility="collapsed",
)
st.markdown('</div>', unsafe_allow_html=True)

MAX_FILE_SIZE_BYTES = 200 * 1024 * 1024  # 200MB per file

u1, u2 = st.columns([3, 1])
with u1:
    st.markdown('<div class="sec-must">📂 Upload Resumes</div>', unsafe_allow_html=True)
    uploaded_files = st.file_uploader(
        "Upload Resumes",
        type=["pdf", "docx", "txt"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )
    st.markdown('<span style="color:#fde047;font-weight:600;font-size:0.88rem;">200MB per file • PDF, DOCX, TXT</span>', unsafe_allow_html=True)

with u2:
    st.write("")
    st.write("")
    use_folder_samples = st.checkbox("Include `resumes/` folder", value=not bool(uploaded_files))

run_btn = st.button("🚀 Screen & Rank Candidates", type="primary")

if run_btn:
    must_have = [s.strip().lower() for s in must_raw.split(",") if s.strip()]
    nice_have = [s.strip().lower() for s in nice_raw.split(",") if s.strip()]
    cleaned_jd = clean_text(jd_desc)

    tmp_dir = Path("outputs/_streamlit_uploads")
    tmp_dir.mkdir(parents=True, exist_ok=True)

    candidate_sources = []
    oversized_files = []

    if uploaded_files:
        for up in uploaded_files:
            if up.size > MAX_FILE_SIZE_BYTES:
                oversized_files.append(up.name)
                continue
            p = tmp_dir / up.name
            p.write_bytes(up.getbuffer())
            candidate_sources.append((up.name, p))

    if oversized_files:
        st.error(f"Skipped files exceeding 200MB limit: {', '.join(oversized_files)}")

    if use_folder_samples and Path("resumes").exists():
        existing_names = {name for name, _ in candidate_sources}
        for fp in sorted(Path("resumes").iterdir()):
            if fp.is_file() and fp.suffix.lower() in {".pdf", ".docx", ".txt"} and fp.name not in existing_names:
                candidate_sources.append((fp.name, fp))

    if not candidate_sources:
        st.warning("Please upload at least one valid resume (PDF, DOCX, TXT up to 200MB) or enable `resumes/` folder.")
    else:
        parsed_items = []
        cleaned_resumes = []
        for fname, fpath in candidate_sources:
            raw_txt, err = extract_resume_text(fpath)
            c_txt = clean_text(raw_txt) if not err else ""
            cleaned_resumes.append(c_txt)
            parsed_items.append((fname, c_txt, err))

        sims = compute_tfidf_similarities(cleaned_jd, cleaned_resumes)
        results = []

        for (fname, c_txt, err), sim in zip(parsed_items, sims):
            cname = Path(fname).stem.replace("_", " ").title()
            if err:
                results.append({
                    "Candidate": cname,
                    "File": fname,
                    "Match Score": 0.0,
                    "Status": "REJECT",
                    "Must-Have": f"0/{len(must_have)}",
                    "Missing Must-Have": ", ".join(must_have) or "None",
                    "Matched Skills": "",
                    "Exp (Yrs)": 0.0,
                    "TF-IDF Sim": 0.0,
                    "Note": err,
                })
                continue

            sk = evaluate_candidate_skills(c_txt, must_have, nice_have)
            sc = score_candidate(
                tfidf_sim=sim,
                matched_must=sk["matched_must_have"],
                total_must=len(must_have),
                matched_nice=sk["matched_nice_to_have"],
                total_nice=len(nice_have),
                years_exp=sk["years_experience"],
                min_exp_years=min_exp,
                shortlist_threshold=shortlist_th,
                consider_threshold=consider_th,
            )
            missing_str = ", ".join(sk["missing_must_have"]) if sk["missing_must_have"] else "None"
            results.append({
                "Candidate": cname,
                "File": fname,
                "Match Score": sc["match_score"],
                "Status": sc["status"],
                "Must-Have": sc["must_have_coverage"],
                "Missing Must-Have": missing_str,
                "Matched Skills": ", ".join(sk["matched_skills"]) if sk["matched_skills"] else "None",
                "Exp (Yrs)": sk["years_experience"],
                "TF-IDF Sim": sc["tfidf_similarity"],
                "Note": "All must-haves met" if missing_str == "None" else f"Missing: {missing_str}",
            })

        df = pd.DataFrame(results).sort_values(by="Match Score", ascending=False).reset_index(drop=True)
        df.insert(0, "Rank", df.index + 1)

        total_cands = len(df)
        shortlisted_cnt = int((df["Status"] == "SHORTLIST").sum())
        consider_cnt = int((df["Status"] == "CONSIDER").sum())
        top_score = float(df["Match Score"].max()) if total_cands else 0.0

        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.markdown(f'<div class="kpi-card kpi-cyan"><div class="kpi-label">Candidates Screened</div><div class="kpi-val">{total_cands}</div></div>', unsafe_allow_html=True)
        with m2:
            st.markdown(f'<div class="kpi-card kpi-emerald"><div class="kpi-label">Shortlisted</div><div class="kpi-val">{shortlisted_cnt}</div></div>', unsafe_allow_html=True)
        with m3:
            st.markdown(f'<div class="kpi-card kpi-amber"><div class="kpi-label">Consider / Review</div><div class="kpi-val">{consider_cnt}</div></div>', unsafe_allow_html=True)
        with m4:
            st.markdown(f'<div class="kpi-card kpi-pink"><div class="kpi-label">Top Match Score</div><div class="kpi-val">{top_score:.1f}</div></div>', unsafe_allow_html=True)

        tab1, tab2, tab3 = st.tabs(["🏆 Candidate Cards", "📊 Score Charts", "📄 CSV & Full Table"])

        with tab1:
            st.subheader(f"Role: {role_title}")
            for _, row in df.iterrows():
                st_val = row["Status"]
                card_cls = "cand-shortlist" if st_val == "SHORTLIST" else ("cand-consider" if st_val == "CONSIDER" else "cand-reject")
                badge_cls = "b-shortlist" if st_val == "SHORTLIST" else ("b-consider" if st_val == "CONSIDER" else "b-reject")
                gap_badge = (
                    '<span class="badge b-ok">✓ No Must-Have Gaps</span>'
                    if row["Missing Must-Have"] == "None"
                    else f'<span class="badge b-gap">⚠ Gap: {row["Missing Must-Have"]}</span>'
                )
                st.markdown(
                    f"""
                    <div class="cand-card {card_cls}">
                        <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:8px;">
                            <div>
                                <strong style="font-size:1.12rem;color:#ffffff;">#{row['Rank']} {row['Candidate']}</strong>
                                <span style="color:#cbd5e1;font-size:0.86rem;">({row['File']})</span>
                            </div>
                            <div>
                                <span class="badge {badge_cls}">{st_val}</span>
                                <span class="badge b-must">Score: {row['Match Score']:.2f}</span>
                                {gap_badge}
                            </div>
                        </div>
                        <div style="margin-top:0.55rem;font-size:0.92rem;color:#f1f5f9;">
                            <b>Must-Have:</b> {row['Must-Have']} &nbsp;|&nbsp;
                            <b>Exp:</b> {row['Exp (Yrs)']} yrs &nbsp;|&nbsp;
                            <b>TF-IDF Sim:</b> {row['TF-IDF Sim']:.4f}<br/>
                            <b>Matched Skills:</b> {row['Matched Skills']}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        with tab2:
            st.markdown("#### Match Score by Candidate")
            st.bar_chart(df.set_index("Candidate")["Match Score"], color="#38bdf8")
            st.markdown("#### TF-IDF Similarity by Candidate")
            st.bar_chart(df.set_index("Candidate")["TF-IDF Sim"], color="#f472b6")

        with tab3:
            st.dataframe(df, use_container_width=True)
            Path("outputs").mkdir(parents=True, exist_ok=True)
            csv_data = df.to_csv(index=False)
            Path("outputs/screening_report.csv").write_text(csv_data, encoding="utf-8")
            st.download_button(
                "⬇️ Download screening_report.csv",
                data=csv_data.encode("utf-8"),
                file_name="screening_report.csv",
                mime="text/csv",
            )
