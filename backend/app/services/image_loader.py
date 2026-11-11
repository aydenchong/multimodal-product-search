from io import BytesIO

from fastapi import HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError

from backend.app.config import MAX_UPLOAD_BYTES


async def load_uploaded_image(file: UploadFile) -> Image.Image:
    if file.content_type not in {
        "image/jpeg",
        "image/png",
        "image/webp",
    }:
        raise HTTPException(
            status_code=415,
            detail="Only JPEG, PNG and WebP images are supported.",
        )

    try:
        content = await file.read(MAX_UPLOAD_BYTES + 1)

        if len(content) > MAX_UPLOAD_BYTES:
            raise HTTPException(
                status_code=413,
                detail="Image exceeds the 5 MB upload limit.",
            )

        with Image.open(BytesIO(content)) as image:
            if image.format not in {"JPEG", "PNG", "WEBP"}:
                raise HTTPException(
                    status_code=415,
                    detail="Unsupported image format.",
                )

            image.load()

            if image.width * image.height > 20_000_000:
                raise HTTPException(
                    status_code=413,
                    detail="Image dimensions are too large.",
                )

            return image.convert("RGB")

    except (UnidentifiedImageError, OSError, ValueError):
        raise HTTPException(
            status_code=400,
            detail="Invalid or corrupted image.",
        )
    finally:
        await file.close()
