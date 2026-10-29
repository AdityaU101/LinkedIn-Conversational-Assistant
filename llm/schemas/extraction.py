"""Pydantic schemas for ontology-constrained LLM extraction output.

Every item must carry ``evidence``: a verbatim quote from the document. Items
whose quote cannot be located in the source are discarded downstream.
"""
from __future__ import annotations

from typing import Any, Literal, Optional, get_args, get_origin

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, ValidationError, model_validator


class _Item(BaseModel):
    model_config = ConfigDict(extra="ignore")
    evidence: str = Field(min_length=2, description="verbatim quote from the document supporting this item")


class _Container(BaseModel):
    """Keeps every valid item and drops invalid ones one by one (e.g. an item without its evidence quote,
    which grounding would discard anyway) instead of rejecting the whole extraction. ``invalid_items``
    counts what was dropped."""
    model_config = ConfigDict(extra="ignore")
    _invalid: int = PrivateAttr(default=0)

    @model_validator(mode="wrap")
    @classmethod
    def _drop_invalid_items(cls, data: Any, handler):
        dropped = 0
        if isinstance(data, dict):
            data = dict(data)
            for name, f in cls.model_fields.items():
                args = get_args(f.annotation)
                if get_origin(f.annotation) is list and args and isinstance(args[0], type) \
                        and issubclass(args[0], BaseModel) and name in data:
                    raw = data[name] if isinstance(data[name], list) else []
                    keep = []
                    for item in raw:
                        try:
                            args[0].model_validate(item)
                            keep.append(item)
                        except ValidationError:
                            dropped += 1
                    data[name] = keep
        out = handler(data)
        out._invalid = dropped
        return out

    @property
    def invalid_items(self) -> int:
        return self._invalid


class LLMOccupation(_Item):
    title: str


class LLMSkill(_Item):
    name: str


class LLMExperience(_Item):
    title: Optional[str] = None
    company: Optional[str] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None


class LLMEducation(_Item):
    degree: Optional[str] = None
    field: Optional[str] = None
    institution: Optional[str] = None
    year: Optional[str] = None


class LLMCredential(_Item):
    name: str
    issuer: Optional[str] = None


class LLMProject(_Item):
    name: str
    description: Optional[str] = None


class ResumeExtraction(_Container):
    occupations: list[LLMOccupation] = Field(default_factory=list)
    skills: list[LLMSkill] = Field(default_factory=list)
    experiences: list[LLMExperience] = Field(default_factory=list)
    education: list[LLMEducation] = Field(default_factory=list)
    credentials: list[LLMCredential] = Field(default_factory=list)
    projects: list[LLMProject] = Field(default_factory=list)


class LLMRequirement(_Item):
    kind: Literal["skill", "experience", "education", "credential", "other"]
    importance: Literal["required", "preferred"] = "required"
    text: str
    years: Optional[float] = None


class JobExtraction(_Container):
    job_title: Optional[str] = None
    occupation: Optional[str] = None
    company: Optional[str] = None
    location: Optional[str] = None
    requirements: list[LLMRequirement] = Field(default_factory=list)
    responsibilities: list[_Item] = Field(default_factory=list)


class SPOTriple(_Item):
    subject: str
    predicate: str
    object: str


class SPOExtraction(_Container):
    triples: list[SPOTriple] = Field(default_factory=list)
