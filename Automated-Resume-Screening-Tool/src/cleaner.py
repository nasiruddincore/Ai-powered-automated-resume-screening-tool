import re


def clean_text(raw_text: str) -> str:
    if not raw_text:
        return ""
    text = raw_text.lower()
    text = re.sub(r"\S+@\S+\.\S+", " ", text)
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"[^a-z0-9\s\+\#\.]", " ", text)
    return re.sub(r"\s+", " ", text).strip()
