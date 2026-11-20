"""Extract text from PDF, DOCX, and TXT resume files."""

from pathlib import Path
from typing import Tuple
import pdfplumber
from docx import Document


def extract_text_from_pdf(file_path: Path) -> Tuple[str, str]:
    """Extract text from a PDF file using pdfplumber."""
    pages_text = []
    try:
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                content = page.extract_text()
                if content:
                    pages_text.append(content)
        full_text = "\n".join(pages_text).strip()
        if not full_text:
            return "", "Empty text extracted (PDF may be scanned/image-based)"
        return full_text, ""
    except Exception as exc:
        return "", f"PDF extraction error: {exc}"


def extract_text_from_docx(file_path: Path) -> Tuple[str, str]:
    """Extract text from a DOCX file including paragraphs and table cells."""
    lines = []
    try:
        doc = Document(file_path)
        for para in doc.paragraphs:
            if para.text and para.text.strip():
                lines.append(para.text.strip())
        for table in doc.tables:
            for row in table.rows:
                row_vals = [cell.text.strip() for cell in row.cells if cell.text and cell.text.strip()]
                if row_vals:
                    lines.append(" | ".join(row_vals))
        full_text = "\n".join(lines).strip()
        if not full_text:
            return "", "Empty text extracted from DOCX"
        return full_text, ""
    except Exception as exc:
        return "", f"DOCX extraction error: {exc}"


def extract_text_from_txt(file_path: Path) -> Tuple[str, str]:
    """Extract text from a TXT file safely."""
    try:
        full_text = file_path.read_text(encoding="utf-8", errors="ignore").strip()
        if not full_text:
            return "", "TXT file is empty"
        return full_text, ""
    except Exception as exc:
        return "", f"TXT extraction error: {exc}"


def extract_resume_text(file_path: Path) -> Tuple[str, str]:
    """Dispatch extraction based on file extension."""
    suffix = file_path.suffix.lower()
    if suffix == ".pdf":
        return extract_text_from_pdf(file_path)
    if suffix == ".docx":
        return extract_text_from_docx(file_path)
    if suffix == ".txt":
        return extract_text_from_txt(file_path)
    return "", f"Unsupported file format: {suffix}"