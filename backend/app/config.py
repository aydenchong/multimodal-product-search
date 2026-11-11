
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = PROJECT_ROOT / "models" / "resnet18_baseline.pth"

FAISS_INDEX_PATH = (
    PROJECT_ROOT / "data" / "processed" / "clip_full_faiss.index"
)

PRODUCT_METADATA_PATH = (
    PROJECT_ROOT / "data" / "processed" / "clip_full_metadata.csv"
)

CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"

PRODUCT_IMAGE_DATASET = "ashraq/fashion-product-images-small"

MAX_UPLOAD_BYTES = 5 * 1024 * 1024
DEFAULT_TOP_K = 5
MAX_TOP_K = 20

CLASS_NAMES = [
    "Apparel",
    "Accessories",
    "Footwear",
    "Personal Care",
]
