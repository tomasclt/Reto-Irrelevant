import re

_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]")


def sanitize_text(value: object, max_length: int = 250) -> str:
    text = "" if value is None else str(value)
    text = _CONTROL_CHARS.sub("", text).strip()
    return text[:max_length]


def sanitize_question(question: str, max_length: int) -> str:
    return sanitize_text(question, max_length=max_length)
