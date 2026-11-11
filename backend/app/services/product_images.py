from io import BytesIO

from datasets import load_dataset
from PIL import Image

from backend.app.config import PRODUCT_IMAGE_DATASET


class ProductImageService:
    """Read-only product image access backed by the existing dataset IDs.

    The dataset is loaded once during application startup. Requests only use
    integer product IDs from the retrieval metadata; no filesystem path is
    accepted from the client.
    """

    def __init__(self):
        self.dataset = load_dataset(PRODUCT_IMAGE_DATASET, split="train")
        self.id_to_index = {
            int(product_id): index
            for index, product_id in enumerate(self.dataset["id"])
        }

    def has_product(self, product_id: int) -> bool:
        return product_id in self.id_to_index

    def jpeg_bytes(self, product_id: int) -> bytes | None:
        index = self.id_to_index.get(product_id)
        if index is None:
            return None

        image = self.dataset[index]["image"].convert("RGB")
        buffer = BytesIO()
        image.save(buffer, format="JPEG", quality=88, optimize=True)
        return buffer.getvalue()
