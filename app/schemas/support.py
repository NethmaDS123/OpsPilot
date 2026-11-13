"""HTTP schemas for policy RAG endpoints."""

from pydantic import BaseModel, Field

from app.schemas.knowledge import SourceReference, SupportAnswer


class SupportQuestionRequest(BaseModel):
    question: str = Field(min_length=3)
    company_id: str = Field(min_length=2)


class SupportQuestionResponse(SupportAnswer):
    pass


class PolicySearchRequest(BaseModel):
    query: str = Field(min_length=2)
    company_id: str = Field(min_length=2)
    top_k: int = Field(default=5, ge=1, le=20)
