
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor

MODEL_NAME = "openai/clip-vit-base-patch32"
BATCH_SIZE = 8

OUTPUT_DIR = Path("data/processed")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_CSV = OUTPUT_DIR / "train.csv"
EMBEDDINGS_PATH = OUTPUT_DIR / "clip_full_embeddings.npy"
METADATA_PATH = OUTPUT_DIR / "clip_full_metadata.csv"

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)

# 1. Load ALL training products
train_df = pd.read_csv(TRAIN_CSV)

print("Training products:", len(train_df))

dataset = load_dataset(
    "ashraq/fashion-product-images-small",
    split="train",
)

id_to_index = {
    int(product_id): i
    for i, product_id in enumerate(dataset["id"])
}

# 2. Load CLIP once
processor = CLIPProcessor.from_pretrained(MODEL_NAME)
model = CLIPModel.from_pretrained(MODEL_NAME)

model = model.to(device)
model.eval()

all_embeddings = []
all_metadata = []

# 3. Extract embeddings in batches
for start in tqdm(
    range(0, len(train_df), BATCH_SIZE),
    desc="Extracting CLIP embeddings",
):
    batch_df = train_df.iloc[start:start + BATCH_SIZE]

    images = []
    metadata_batch = []

    for row in batch_df.itertuples():
        product = dataset[id_to_index[int(row.id)]]

        images.append(
            product["image"].convert("RGB")
        )

        metadata_batch.append({
            "id": int(row.id),
            "name": product["productDisplayName"],
            "category": product["masterCategory"],
            "article_type": product["articleType"],
        })

    inputs = processor(
        images=images,
        return_tensors="pt",
    )

    pixel_values = inputs["pixel_values"].to(device)

    with torch.inference_mode():
        features = model.get_image_features(
            pixel_values=pixel_values
        )

    if not isinstance(features, torch.Tensor):
        features = features.pooler_output

    features = torch.nn.functional.normalize(
        features,
        p=2,
        dim=-1,
    )

    embeddings = (
        features.cpu().numpy().astype("float32")
    )

    all_embeddings.append(embeddings)
    all_metadata.extend(metadata_batch)

# 4. Save complete embedding matrix
embedding_matrix = np.concatenate(
    all_embeddings,
    axis=0,
)

metadata_df = pd.DataFrame(all_metadata)

assert len(embedding_matrix) == len(train_df)
assert len(metadata_df) == len(train_df)
assert np.isfinite(embedding_matrix).all()
assert metadata_df["id"].is_unique

np.save(
    EMBEDDINGS_PATH,
    embedding_matrix,
)

metadata_df.to_csv(
    METADATA_PATH,
    index=False,
)

print("\nExtraction Complete!")
print("Embeddings shape:", embedding_matrix.shape)
print("Metadata rows:", len(metadata_df))
print("Unique article types:", metadata_df["article_type"].nunique())
print("Saved:", EMBEDDINGS_PATH)
print("Saved:", METADATA_PATH)
