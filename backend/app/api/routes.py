import time

from fastapi import APIRouter, File, HTTPException, Query, Request, UploadFile
from fastapi.responses import StreamingResponse
from starlette.concurrency import run_in_threadpool

from backend.app.config import DEFAULT_TOP_K, MAX_TOP_K
from backend.app.schemas.product import AnalysisResponse
from backend.app.services.image_loader import load_uploaded_image


router = APIRouter(prefix="/api/v1", tags=["Products"])


def run_analysis(request: Request, image, top_k: int):
    classification = request.app.state.classifier.predict(image)

    similar_products = request.app.state.retriever.search(
        image,
        top_k=top_k,
    )

    return classification, similar_products


@router.post("/analyze", response_model=AnalysisResponse)
async def analyze_product(
    request: Request,
    file: UploadFile = File(...),
    top_k: int = Query(
        default=DEFAULT_TOP_K,
        ge=1,
        le=MAX_TOP_K,
    ),
):
    start = time.perf_counter()

    image = await load_uploaded_image(file)

    if not request.app.state.models_ready:
        raise HTTPException(
            status_code=503,
            detail="ML models are not ready.",
        )

    classification, similar_products = await run_in_threadpool(
        run_analysis,
        request,
        image,
        top_k,
    )

    product_images = getattr(request.app.state, "product_images", None)
    if product_images is not None:
        similar_products = [
            {
                **product,
                "image_url": (
                    f"/api/v1/products/{product['id']}/image"
                    if product_images.has_product(int(product["id"]))
                    else None
                ),
            }
            for product in similar_products
        ]

    elapsed_ms = (time.perf_counter() - start) * 1000

    return {
        "classification": classification,
        "similar_products": similar_products,
        "processing_time_ms": round(elapsed_ms, 2),
    }


@router.get("/products/{product_id}/image", response_class=StreamingResponse)
def get_product_image(product_id: int, request: Request):
    product_images = getattr(request.app.state, "product_images", None)
    if product_images is None:
        raise HTTPException(status_code=404, detail="Product image dataset is unavailable.")

    image_bytes = product_images.jpeg_bytes(product_id)
    if image_bytes is None:
        raise HTTPException(status_code=404, detail="Product image was not found.")

    return StreamingResponse(
        iter([image_bytes]),
        media_type="image/jpeg",
        headers={"Cache-Control": "public, max-age=86400"},
    )
