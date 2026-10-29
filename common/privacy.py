"""PII redaction, PII-aware logging and prompt-injection screening."""
from __future__ import annotations

import logging
import re

EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
PHONE = re.compile(r"(?<!\d)(?:\+?\d{1,3}[\s.-]?)?(?:\(\d{3}\)|\d{3})[\s.-]?\d{3}[\s.-]?\d{4}(?!\d)")
URL = re.compile(r"\bhttps?://\S+|\bwww\.\S+", re.I)
SSN = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
STREET = re.compile(r"\b\d{1,5}\s+(?:[A-Z][a-z]+\s){1,3}(?:Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Lane|Ln|Drive|Dr|Court|Ct)\b\.?")


def redact(text: str) -> str:
    """Replace direct identifiers with typed placeholders."""
    if not text:
        return text
    text = EMAIL.sub("[EMAIL]", text)
    text = SSN.sub("[SSN]", text)
    text = PHONE.sub("[PHONE]", text)
    text = URL.sub("[URL]", text)
    text = STREET.sub("[ADDRESS]", text)
    return text


def find_pii(text: str) -> dict[str, int]:
    return {k: len(p.findall(text or "")) for k, p in
            {"email": EMAIL, "phone": PHONE, "url": URL, "ssn": SSN, "street_address": STREET}.items()}


class PIIRedactingFilter(logging.Filter):
    """Logging filter that redacts identifiers from every record's message."""

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            msg = record.getMessage()
        except Exception:
            return True
        record.msg = redact(msg)
        record.args = ()
        return True


def configure_logging(level: str = "INFO") -> None:
    root = logging.getLogger()
    if not any(isinstance(f, PIIRedactingFilter) for h in root.handlers for f in h.filters):
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
        handler.addFilter(PIIRedactingFilter())
        root.addHandler(handler)
    root.setLevel(level)


INJECTION_PATTERNS = [
    r"ignore (?:all |any )?(?:the )?(?:previous|prior|above) (?:instructions|prompts|rules)",
    r"disregard (?:all |any )?(?:previous|prior|above|the) (?:instructions|rules)",
    r"you are now (?:an?|the|in)\b", r"new instructions\s*:", r"system prompt",
    # 'act as an AI…' but not job titles like 'act as Assistant Sales Manager' (capitalised next word)
    r"\bact as (?:an? |the )?(?:ai|assistant|language model|chatbot|llm|system)\b(?![ -]+(?-i:[A-Z]))",
    r"(?:rate|score|rank) (?:this|me|the candidate) (?:as )?(?:highly|10/10|the best|first)",
    r"</?(?:system|assistant|instructions?)>", r"\[\s*(?:system|inst)\s*\]",
    r"do not (?:mention|reveal|tell) (?:the |this to the )?(?:user|recruiter|instructions|prompt)",
    r"respond only with", r"output only the following",
    r"(?:match|recommend) (?:this|me) (?:to|for) (?:every|all) ",
]
_INJ = re.compile("|".join(INJECTION_PATTERNS), re.I)


def detect_injection(text: str) -> list[str]:
    """Return suspicious instruction-like phrases found in untrusted document text."""
    return sorted({m.group(0) for m in _INJ.finditer(text or "")})


def neutralize(text: str) -> str:
    """Defang suspicious phrases and delimiter tags so they read as quoted data."""
    text = re.sub(r"</?(document|system|assistant|instructions?)>", lambda m: m.group(0).replace("<", "‹").replace(">", "›"),
                  text or "", flags=re.I)
    return _INJ.sub(lambda m: f"«{m.group(0)}»", text)


def wrap_untrusted(text: str, label: str = "document") -> str:
    """Delimit untrusted content. Instructions inside it must be treated as data."""
    return f"<{label}>\n{neutralize(text)}\n</{label}>"
