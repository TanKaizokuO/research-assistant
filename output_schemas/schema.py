from pydantic import BaseModel, Field, field_validator, model_validator, HttpUrl
from typing import Literal
from datetime import datetime, timezone


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
        ...,
        min_length=20,
        max_length=2000,  # FIX: was silently truncating; consider raising if content is
        description="Short abstract or snippet",  # longer — or increase this limit
    )

    confidence_score: int = Field(
        ...,
        ge=0,
        le=100,
        description="Relevance/confidence score (0–100)",
        # FIX: description said "0-10" but constraint allowed 0–100. Aligned description
        # to match the actual constraint. If you meant 0–10, change `le=100` to `le=10`.
    )

    # --- Validators ---

    @field_validator("source_id")
    @classmethod
    def validate_source_id(cls, v: str) -> str:
        if " " in v:
            raise ValueError("source_id must not contain spaces")
        return v.lower()

    @field_validator("title")
    @classmethod
    def clean_title(cls, v: str) -> str:
        cleaned = v.strip()
        # FIX: strip() can reduce length below min_length (e.g. "     hi" -> "hi").
        # Validate the post-strip length explicitly since Pydantic's min_length
        # check runs on the raw value, before this validator fires.
        if len(cleaned) < 5:
            raise ValueError(
                "title must be at least 5 characters after stripping whitespace"
            )
        return cleaned

    @field_validator("summary")
    @classmethod
    def clean_summary(cls, v: str) -> str:
        cleaned = v.strip()
        # FIX: same strip-then-validate issue as title above.
        if len(cleaned) < 20:
            raise ValueError(
                "summary must be at least 20 characters after stripping whitespace"
            )
        return cleaned

    @field_validator("retrieval_timestamp")
    @classmethod
    def validate_timestamp_not_future(cls, v: datetime) -> datetime:
        # FIX: was missing entirely. Without this, a typo or bad data source
        # can produce timestamps years in the future with no error raised.
        now = datetime.now(tz=timezone.utc)
        aware = v if v.tzinfo is not None else v.replace(tzinfo=timezone.utc)
        if aware > now:
            raise ValueError(
                f"retrieval_timestamp cannot be in the future (got {v.isoformat()})"
            )
        return v

    @model_validator(mode="after")
    def validate_pdf_url(self) -> "SourceSchema":
        # FIX: was missing entirely. source_type="pdf" should have a .pdf URL,
        # and non-pdf source types should not point to a PDF file, catching
        # mismatches between the URL and the declared type.
        url_str = str(self.url).lower()
        if self.source_type == "pdf" and not url_str.endswith(".pdf"):
            raise ValueError("source_type is 'pdf' but url does not end with .pdf")
        if self.source_type != "pdf" and url_str.endswith(".pdf"):
            raise ValueError(
                f"url ends with .pdf but source_type is '{self.source_type}' — "
                "set source_type='pdf' or use a non-PDF url"
            )
        return self
