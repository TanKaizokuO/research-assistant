from pydantic import BaseModel, HttpUrl, Field, field_validator
from datetime import datetime
from typing import Literal


class SourceSchema(BaseModel):
    source_id: str = Field(
        ...,
        min_length=3,
        max_length=100,
        description="Unique identifier for the source",
    )

    url: HttpUrl = Field(..., description="Valid URL of the source")

    title: str = Field(
        ..., min_length=5, max_length=300, description="Title of the source"
    )

    source_type: Literal["web_article", "pdf", "academic_paper", "blog", "report"] = (
        Field(..., description="Type of the source")
    )

    retrieval_timestamp: datetime = Field(
        ..., description="ISO-8601 timestamp of retrieval"
    )

    summary: str = Field(
        ..., min_length=20, max_length=2000, description="Short abstract or snippet"
    )

    confidence_score: int = Field(
        ..., ge=0, le=10, description="Relevance/confidence score (0-10)"
    )

    # --- Validators ---

    @field_validator("source_id")
    @classmethod
    def validate_source_id(cls, v: str) -> str:
        if " " in v:
            raise ValueError("source_id must not contain spaces")
        return v.lower()

    @field_validator("summary")
    @classmethod
    def clean_summary(cls, v: str) -> str:
        return v.strip()

    @field_validator("title")
    @classmethod
    def clean_title(cls, v: str) -> str:
        return v.strip()
