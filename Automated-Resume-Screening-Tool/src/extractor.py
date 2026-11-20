from pathlib import Path
from typing import Tuple
import pdfplumber
from docx import Document


def extract_resume_text(file_path: Path) -> Tuple[str, str]:
    suffix = file_path.suffix.lower()
    try:
        if suffix == ".pdf":
            pages = []
            with pdfplumber.open(file_path) as pdf:
                for p in pdf.pages:
                    t = p.extract_text()
                    if t:
                        pages.append(t)
            txt = "\n".join(pages).strip()
            return (txt, "") if txt else ("", "Empty PDF text")
        if suffix == ".docx":
            doc = Document(file_path)
            lines = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
            txt = "\n".join(lines).strip()
            return (txt, "") if txt else ("", "Empty DOCX text")
        if suffix == ".txt":
            txt = file_path.read_text(encoding="utf-8", errors="ignore").strip()
            return (txt, "") if txt else ("", "Empty TXT file")
        return "", f"Unsupported extension: {suffix}"
    except Exception as exc:
        return "", f"Extraction error: {exc}"
