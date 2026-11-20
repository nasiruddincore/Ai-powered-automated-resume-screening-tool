# 🎯 AI-Powered Automated Resume Screening Tool

**Smart NLP & TF-IDF Candidate Ranker with Must-Have Gap Detection, Multi-Color Streamlit Command Center & CSV Reporting**

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-TF--IDF%20%26%20NLP-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)
[![Pandas](https://img.shields.io/badge/Pandas-Data%20Pipeline-150458?style=for-the-badge&logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![NumPy](https://img.shields.io/badge/NumPy-Vector%20Math-013243?style=for-the-badge&logo=numpy&logoColor=white)](https://numpy.org/)
[![PowerShell](https://img.shields.io/badge/Windows-PowerShell%20Ready-5391FE?style=for-the-badge&logo=powershell&logoColor=white)](https://learn.microsoft.com/powershell/)
[![License: MIT](https://img.shields.io/badge/License-MIT-10B981?style=for-the-badge)](https://opensource.org/licenses/MIT)

</div>

---

## 📌 Overview

The **AI-Powered Automated Resume Screening Tool** (`ai-powered-automated-resume-screening-tool`) is an industry-oriented Python & NLP pipeline that automates initial candidate screening and ranking. It ingests resumes in **PDF**, **DOCX**, and **TXT** formats, normalizes text, extracts technical skills and years of experience, computes **TF-IDF cosine similarity** against a Job Description, flags missing **must-have skills**, and generates ranked CSV reports alongside an interactive multi-color **Streamlit** dashboard.

### ✨ Key Features
- 📄 **Multi-Format Resume Ingestion:** Extracts text cleanly from `.pdf` (`pdfplumber`), `.docx` (`python-docx`), and `.txt` files (up to `200MB` per file in UI).
- 🧹 **Regex Text Normalization:** Strips emails, URLs, and formatting noise while preserving technical tokens (`c++`, `c#`, `power bi`, `scikit-learn`).
- 🧠 **Skill & Experience Extraction:** Maps skill aliases to canonical names and detects years of experience automatically.
- 📊 **Hybrid NLP + Rule Scoring:** Blends **TF-IDF cosine similarity** with **must-have coverage**, **nice-to-have skills**, **experience fit**, and **missing must-have penalties**.
- 🚨 **Explainable Gap Flagging:** Classifies candidates into `SHORTLIST`, `CONSIDER`, or `REJECT` and explicitly highlights any missing must-have skills.
- 🎨 **Multi-Color Streamlit Command Center:** High-contrast dark UI with KPI cards, color-coded candidate badges, score bar charts, and one-click CSV download.

---

## 🏗️ Project Architecture & Folder Structure

```text
ai-powered-automated-resume-screening-tool/
│
├── .streamlit/
│   └── config.toml           # Custom high-contrast dark multi-color UI theme
├── resumes/                  # Input sample & demo resumes (.pdf, .docx, .txt)
├── data/
│   └── job_description.json  # Configurable JD, must-have & nice-to-have skills, thresholds
├── src/
│   ├── __init__.py
│   ├── extractor.py          # PDF, DOCX, and TXT text extraction
│   ├── cleaner.py            # Regex text cleaning and normalization
│   ├── matcher.py            # Skill alias matching & years-of-experience estimator
│   └── scorer.py             # TF-IDF vectorization, cosine similarity & weighted scoring
├── outputs/
│   └── screening_report.csv  # Auto-generated ranked screening report
├── images/                   # Project & dashboard screenshots for GitHub proof of work
├── docs/                     # Architecture notes and simulation documentation
├── app.py                    # Multi-color Streamlit recruiter dashboard
├── main.py                   # CLI batch screening & CSV report generator
├── requirements.txt          # Python package dependencies
├── .gitignore                # Git ignore rules (.venv, __pycache__, temp uploads)
└── README.md                 # Project documentation
🛠️ Tech StackLayerTechnologies & LibrariesCore LanguageNLP & ML Scoring re (Regex)Data Processing Document Parsingpdfplumber (PDF), python-docx (DOCX), UTF-8 TextWeb Dashboard⚡ Quick Start (Windows PowerShell)1️⃣ Clone the RepositoryPowerShellgit clone [https://github.com/](https://github.com/)<your-username>/ai-powered-automated-resume-screening-tool.git
cd ai-powered-automated-resume-screening-tool
2️⃣ Create & Activate Virtual EnvironmentPowerShellpython -m venv .venv
.venv\Scripts\Activate.ps1
(If script execution is restricted in PowerShell, run: Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass first, or use .venv\Scripts\activate.)3️⃣ Install DependenciesPowerShellpython -m pip install --upgrade pip
pip install -r requirements.txt
4️⃣ Run the CLI Screening PipelinePowerShellpython main.py
This processes all resumes in resumes/, prints the ranked leaderboard in PowerShell, and exports outputs\screening_report.csv.5️⃣ Launch the Multi-Color Streamlit DashboardPowerShellpython -m streamlit run app.py
💡 Windows Tip: Using python -m streamlit run app.py runs Streamlit directly through your approved python.exe interpreter and avoids Windows Application Control blocks on streamlit.exe inside .venv\Scripts\.🧪 Default Job Criteria & Demo SimulationThe tool ships pre-configured for Junior Data Analyst / Python Automation:🔥 Must-Have Skills: python, sql, pandas, excel✨ Nice-to-Have Skills: numpy, power bi, tableau, scikit-learn, git, data visualization, statistics📝 Job Description: Looking for a detail-oriented candidate with strong Python, SQL, Pandas, and Excel skills to clean data, automate recurring reports, build dashboards, and support analytics workflows. Experience with Power BI, Tableau, NumPy, scikit-learn, and Git is a plus.Sample CLI OutputPlaintextrank         candidate_name  match_score    status must_have_coverage             missing_must_have
   1  Demo 01 Aarav Mehta Strong        92.40 SHORTLIST                4/4                          None
   2 Demo 02 Neha Kapoor Partial        56.80  CONSIDER                2/4                python, pandas
   3  Demo 03 Kabir Verma Low Match     11.20    REJECT                0/4   python, sql, pandas, excel
📸 ScreenshotsAdd your captured screenshots inside images/ to showcase the workflow:images/dashboard_ui.png — Multi-Color Streamlit Command Center & Skill Badgesimages/candidate_cards.png — Ranked Candidate Scorecards & Must-Have Gap Flagsimages/terminal_output.png — PowerShell CLI Screening Outputimages/csv_report.png — Generated outputs/screening_report.csv🚀 Future EnhancementsSemantic dense embeddings with sentence-transformers (all-MiniLM-L6-v2)Fuzzy skill matching with rapidfuzz and entity extraction with spaCyFastAPI REST endpoints and SQLite/PostgreSQL persistence for ATS integration
'@ | Set-Content -Path "README.md" -Encoding UTF8
