# Automated Resume Screening Tool

An industry-aligned Python NLP and data pipeline that ingests resumes (PDF, DOCX, TXT), cleans text, extracts skills and experience, computes TF-IDF cosine similarity against a job description, flags must-have skill gaps, ranks candidates, and exports a CSV screening report (with an optional Streamlit UI).

## Problem Statement
Recruiters often review hundreds of resumes per role manually, leading to slow shortlisting and inconsistent initial filtering. This project automates early-stage screening using transparent skill rules and text similarity so recruiters can focus on qualified candidates and inspect exact skill gaps.

## Industry Relevance
Modern HR tech and ATS workflows rely on structured skill extraction, requirement coverage checks, text similarity scoring, and auditable outputs. This project demonstrates core **Python**, **Data Analysis**, **NLP/IR**, and **Automation** skills on realistic synthetic resumes and job descriptions.

## Features
- Multi-format resume extraction (`.pdf`, `.docx`, `.txt`) with error handling.
- Regex-based text cleaning and normalization.
- Curated skill matching (`must_have` and `nice_to_have`) plus years-of-experience estimation.
- TF-IDF vectorization + cosine similarity combined with must-have skill coverage and gap penalties.
- Candidate ranking (`SHORTLIST`, `CONSIDER`, `REJECT`) and CSV report export (`outputs/screening_report.csv`).
- Optional interactive Streamlit dashboard (`app.py`).

## Tech Stack
- **Language:** Python 3.10+
- **Data & Math:** `pandas`, `numpy`
- **NLP & Similarity:** `re` (Regex), `scikit-learn` (`TfidfVectorizer`, `cosine_similarity`)
- **Document Parsing:** `pdfplumber`, `python-docx`
- **Optional UI:** `streamlit`

## Folder Structure
```text
Automated-Resume-Screening-Tool/
├── resumes/
├── data/
├── src/
├── outputs/
├── images/
├── docs/
├── app.py
├── main.py
├── requirements.txt
├── .gitignore
└── README.md
