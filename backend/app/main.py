
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request

from backend.app.api.routes import router
from backend.app.services.classification import ClassificationService
from backend.app.services.retrieval import RetrievalService


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.models_ready = False

    print("Loading ShopSense AI models...")

    app.state.classifier = ClassificationService()
    app.state.retriever = RetrievalService()

    app.state.models_ready = True

    print("ShopSense AI models loaded successfully.")

    yield

    app.state.models_ready = False
    app.state.classifier = None
    app.state.retriever = None


app = FastAPI(
    title="ShopSense AI API",
    description="Multimodal E-Commerce Product Intelligence Platform",
    version="0.2.0",
    lifespan=lifespan,
)

app.include_router(router)


@app.get("/health", tags=["System"])
def health_check():
    return {
        "status": "healthy",
        "service": "shopsense-ai-backend",
        "version": "0.2.0",
    }


@app.get("/ready", tags=["System"])
def readiness_check(request: Request):
    ready = getattr(request.app.state, "models_ready", False)

    return {
        "ready": ready,
        "models": ["resnet18", "clip", "faiss"] if ready else [],
    }
