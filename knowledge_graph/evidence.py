"""Evidence nodes + provenance checks."""
from __future__ import annotations

from common.schemas import Evidence


def evidence_node(ev: Evidence) -> dict:
    return {"id": ev.evidence_id, "document_id": ev.source_document_id, "text": ev.source_text[:1000],
            "start": ev.start_offset, "end": ev.end_offset, "extractor": ev.extractor,
            "extractor_version": ev.extractor_version, "confidence": ev.confidence}


def verify_offsets(ev: Evidence, document_text: str) -> bool:
    """True when the evidence text is exactly the referenced span of the document."""
    return document_text[ev.start_offset:ev.end_offset] == ev.source_text
