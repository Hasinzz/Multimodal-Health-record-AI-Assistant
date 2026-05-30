import re


def clean_ocr_text(text: str) -> str:
    if text is None:
        return ""

    text = text.replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[_|]{2,}", " ", text)
    text = re.sub(r"[^\S\n]+", " ", text)

    lines = []

    for line in text.splitlines():
        line = line.strip()

        if not line:
            continue

        if len(line) == 1 and not line.isalnum():
            continue

        lines.append(line)

    return "\n".join(lines)


def make_preview(text: str, limit: int = 800) -> str:
    text = clean_ocr_text(text)

    if len(text) <= limit:
        return text

    return text[:limit] + "..."