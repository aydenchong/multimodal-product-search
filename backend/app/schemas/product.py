from pydantic import BaseModel, Field


class ClassificationResult(BaseModel):
    category: str
    confidence: float = Field(ge=0, le=1)


class SimilarProduct(BaseModel):
    id: int
    name: str
    category: str
    article_type: str
    similarity: float
    image_url: str | None = None


class AnalysisResponse(BaseModel):
    classification: ClassificationResult
    similar_products: list[SimilarProduct]
    processing_time_ms: float
