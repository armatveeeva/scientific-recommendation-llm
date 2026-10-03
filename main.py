from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import ml_service

app = FastAPI(
    title="Scientific Articles Recommendation API",
    description="Система рекомендаций научных публикаций с использованием векторного поиска и LLM-суммаризации",
    version="1.0.0"
)

class QueryRequest(BaseModel):
    text: str
    top_k: int = 2

class RecommendationResponse(BaseModel):
    similarity: float
    title: str
    abstract: str
    summary: str

@app.post("/recommend", response_model=List[RecommendationResponse])
async def get_recommendations(request: QueryRequest):
    try:
        results = ml_service.find_similar_articles(request.text, request.top_k)
        response = []
        for res in results:
            summary = ml_service.summarize_text(res["abstract"])
            response.append(RecommendationResponse(
                similarity=res["similarity"],
                title=res["title"],
                abstract=res["abstract"],
                summary=summary
            ))
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "ok"}
