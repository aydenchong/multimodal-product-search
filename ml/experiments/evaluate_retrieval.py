from pathlib import Path

import faiss
import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from transformers import CLIPModel, CLIPProcessor

MODEL_NAME = "openai/clip-vit-base-patch32"
NUM_QUERIES = 200
BATCH_SIZE = 8
TOP_K = 5
SEED = 42

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)

# 1. Load saved FAISS index and product metadata
index = faiss.read_index(
    "data/processed/clip_faiss.index"
)

metadata = pd.read_csv(
    "data/processed/clip_product_metadata.csv"
)

assert index.ntotal == len(metadata)

# 2. Load validation products
val_df = pd.read_csv("data/processed/val.csv")

selected = val_df.sample(
    n=min(NUM_QUERIES, len(val_df)),
    random_state=SEED,
).reset_index(drop=True)

dataset = load_dataset(
    "ashraq/fashion-product-images-small",
    split="train",
)

id_to_index = {
    int(product_id): i
    for i, product_id in enumerate(dataset["id"])
}

# 3. Load pretrained CLIP
processor = CLIPProcessor.from_pretrained(MODEL_NAME)
model = CLIPModel.from_pretrained(MODEL_NAME)

model = model.to(device)
model.eval()

results = []

# 4. Process validation queries in batches
for start in range(0, len(selected), BATCH_SIZE):
    batch_df = selected.iloc[start:start + BATCH_SIZE]

    images = []
    query_records = []

    for row in batch_df.itertuples():
        sample = dataset[id_to_index[int(row.id)]]

        images.append(
            sample["image"].convert("RGB")
        )

        query_records.append({
            "id": int(row.id),
            "name": sample["productDisplayName"],
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

    if not isinstance(features, torch.Tensor):
        features = features.pooler_output

    features = torch.nn.functional.normalize(
        features,
        p=2,
        dim=-1,
    )

    query_vectors = features.cpu().numpy().astype(
        "float32"
    )

    # 5. Retrieve Top-K products
    scores, indices = index.search(
        query_vectors,
        TOP_K,
    )

    # 6. Calculate relevance metrics
    for i, query in enumerate(query_records):
        retrieved_indices = indices[i]

        relevant = []

        for result_index in retrieved_indices:
            if result_index < 0:
                relevant.append(0)
                continue

            retrieved_type = metadata.iloc[
                result_index
            ]["article_type"]

            relevant.append(
                int(retrieved_type == query["article_type"])
            )

        precision_at_5 = sum(relevant) / TOP_K
        hit_at_5 = int(any(relevant))
        precision_at_1 = relevant[0]

        reciprocal_rank = 0.0

        for rank, is_relevant in enumerate(
            relevant,
            start=1,
        ):
            if is_relevant:
                reciprocal_rank = 1.0 / rank
                break

        results.append({
            "product_id": query["id"],
            "product_name": query["name"],
            "article_type": query["article_type"],
            "precision_at_1": precision_at_1,
            "precision_at_5": precision_at_5,
            "hit_at_5": hit_at_5,
            "reciprocal_rank_at_5": reciprocal_rank,
        })

    processed = min(
        start + BATCH_SIZE,
        len(selected),
    )

    if processed % 40 == 0:
        print(
            f"Evaluated {processed}/{len(selected)}"
        )

# 7. Aggregate results
results_df = pd.DataFrame(results)

print("\nCLIP + FAISS Retrieval Evaluation")
print("Queries:", len(results_df))
print("Index size:", index.ntotal)

print(
    f"Precision@1: "
    f"{results_df['precision_at_1'].mean():.2%}"
)

print(
    f"Precision@5: "
    f"{results_df['precision_at_5'].mean():.2%}"
)

print(
    f"Hit@5: "
    f"{results_df['hit_at_5'].mean():.2%}"
)

print(
    f"MRR@5: "
    f"{results_df['reciprocal_rank_at_5'].mean():.4f}"
)

# 8. Save evaluation results
output_dir = Path("experiments")
output_dir.mkdir(parents=True, exist_ok=True)

output_path = output_dir / "retrieval_evaluation.csv"

results_df.to_csv(
    output_path,
    index=False,
)

print("Saved:", output_path)
