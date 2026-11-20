"""Clean and normalize extracted resume and job description text."""

import re


def clean_text(raw_text: str) -> str:
    """Lowercase text, remove emails/URLs/noise, and normalize whitespace."""
    if not raw_text:
        return ""

    text = raw_text.lower()
    # Remove email addresses and URLs
    text = re.sub(r"\S+@\S+\.\S+", " ", text)
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    # Preserve letters, numbers, spaces, +, #, and dots (for tools like c++, c#, power bi)
    text = re.sub(r"[^a-z0-9\s\+\#\.]", " ", text)
    # Collapse multiple spaces/newlines into a single space
    text = re.sub(r"\s+", " ", text).strip()
    return text