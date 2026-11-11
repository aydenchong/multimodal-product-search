
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from backend.app.api.routes import router
from backend.app.services.classification import ClassificationService
from backend.app.services.retrieval import RetrievalService
from backend.app.services.product_images import ProductImageService


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.models_ready = False

    print("Loading ShopSense AI models...")

    app.state.classifier = ClassificationService()
    app.state.retriever = RetrievalService()

    try:
        app.state.product_images = ProductImageService()
    except Exception as exc:
        # Image previews are additive: keep inference available if the optional
        # cached image dataset is unavailable in a deployment environment.
        print(f"Product image previews unavailable: {exc}")
        app.state.product_images = None

    app.state.models_ready = True

    print("ShopSense AI models loaded successfully.")

    yield

    app.state.models_ready = False
    app.state.classifier = None
    app.state.retriever = None
    app.state.product_images = None


app = FastAPI(
    title="ShopSense AI API",
    description="Multimodal E-Commerce Product Intelligence Platform",
    version="0.2.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:5174",
        "http://127.0.0.1:5174",
    ],
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
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
