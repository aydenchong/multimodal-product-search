from pathlib import Path

import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from transformers import CLIPModel, CLIPProcessor

MODEL_NAME = "openai/clip-vit-base-patch32"
NUM_PRODUCTS = 1000
BATCH_SIZE = 8
SEED = 42

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)

# 1. Load clean training metadata
train_df = pd.read_csv("data/processed/train.csv")

selected = train_df.sample(
    n=min(NUM_PRODUCTS, len(train_df)),
    random_state=SEED,
).reset_index(drop=True)

# 2. Load original product images
dataset = load_dataset(
    "ashraq/fashion-product-images-small",
    split="train",
)

id_to_index = {
    int(product_id): index
    for index, product_id in enumerate(dataset["id"])
}

# 3. Load pretrained CLIP
processor = CLIPProcessor.from_pretrained(MODEL_NAME)
model = CLIPModel.from_pretrained(MODEL_NAME)

model = model.to(device)
model.eval()

embeddings = []
product_records = []

# 4. Extract image embeddings in batches
for start in range(0, len(selected), BATCH_SIZE):
    batch_df = selected.iloc[start:start + BATCH_SIZE]

    images = []
    batch_records = []

    for row in batch_df.itertuples():
        sample = dataset[id_to_index[int(row.id)]]

        images.append(
            sample["image"].convert("RGB")
        )

        batch_records.append({
            "id": int(row.id),
            "name": sample["productDisplayName"],
            "category": sample["masterCategory"],
            "article_type": sample["articleType"],
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

    # Support model outputs with pooler_output
    if not isinstance(features, torch.Tensor):
        features = features.pooler_output

    # Normalize embeddings for cosine similarity
    features = torch.nn.functional.normalize(
        features,
        p=2,
        dim=-1,
    )

    embeddings.append(
        features.cpu().numpy()
    )

    product_records.extend(batch_records)

    processed = min(start + BATCH_SIZE, len(selected))

    if processed % 100 < BATCH_SIZE:
        print(f"Processed {processed}/{len(selected)}")

# 5. Combine all embedding batches
embedding_matrix = np.concatenate(
    embeddings,
    axis=0,
).astype("float32")

print("\nEmbedding shape:", embedding_matrix.shape)
print("Embedding dtype:", embedding_matrix.dtype)

# 6. Save embeddings and metadata
output_dir = Path("data/processed")
output_dir.mkdir(parents=True, exist_ok=True)

np.save(
    output_dir / "clip_image_embeddings.npy",
    embedding_matrix,
)

pd.DataFrame(product_records).to_csv(
    output_dir / "clip_product_metadata.csv",
    index=False,
)

print("\nSaved:")
print("data/processed/clip_image_embeddings.npy")
print("data/processed/clip_product_metadata.csv")
