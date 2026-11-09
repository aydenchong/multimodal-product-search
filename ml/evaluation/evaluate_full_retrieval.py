from pathlib import Path
import argparse
import faiss
import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor

parser = argparse.ArgumentParser(
    description="Evaluate CLIP + FAISS product retrieval"
)

parser.add_argument(
    "--split",
    choices=["val", "test"],
    default="val",
    help="Dataset split used as retrieval queries",
)

args = parser.parse_args()
MODEL_NAME = "openai/clip-vit-base-patch32"
BATCH_SIZE = 8
TOP_K = 5

DATA_DIR = Path("data/processed")
OUTPUT_DIR = Path("experiments")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

QUERY_PATH = DATA_DIR / f"{args.split}.csv"
METADATA_PATH = DATA_DIR / "clip_full_metadata.csv"
INDEX_PATH = DATA_DIR / "clip_full_faiss.index"
OUTPUT_PATH = (
    OUTPUT_DIR / f"full_retrieval_{args.split}.csv"
)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


print("Device:", device)

# 1. Load complete validation set and gallery
val_df = pd.read_csv(QUERY_PATH)
metadata = pd.read_csv(METADATA_PATH)
index = faiss.read_index(str(INDEX_PATH))

print("Validation queries:", len(val_df))
print("Gallery products:", index.ntotal)

assert len(metadata) == index.ntotal
assert index.d == 512
assert val_df["id"].is_unique
assert metadata["id"].is_unique

# Prevent exact product-ID overlap
overlap = set(val_df["id"]) & set(metadata["id"])
assert not overlap, "Validation IDs overlap with gallery!"

# 2. Calculate gallery category coverage
type_counts = metadata["article_type"].value_counts()

covered_queries = (
    val_df["articleType"].isin(type_counts.index)
)

print(
    "Category coverage:",
    f"{covered_queries.mean():.2%}",
)

# 3. Load original product images
dataset = load_dataset(
    "ashraq/fashion-product-images-small",
    split="train",
)

id_to_index = {
    int(product_id): i
    for i, product_id in enumerate(dataset["id"])
}

# 4. Load CLIP once
processor = CLIPProcessor.from_pretrained(MODEL_NAME)
model = CLIPModel.from_pretrained(MODEL_NAME)

model = model.to(device)
model.eval()

results = []

# 5. Process every validation query
for start in tqdm(
    range(0, len(val_df), BATCH_SIZE),
    desc="Evaluating full validation set",
):
    batch_df = val_df.iloc[start:start + BATCH_SIZE]

    images = []

    for row in batch_df.itertuples():
        product = dataset[id_to_index[int(row.id)]]

        images.append(
            product["image"].convert("RGB")
        )

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

    query_vectors = (
        features.cpu().numpy().astype("float32")
    )

    # 6. Search the complete FAISS gallery
    scores, indices = index.search(
        query_vectors,
        TOP_K,
    )

    # 7. Evaluate each query
    for i, row in enumerate(batch_df.itertuples()):
        retrieved_indices = indices[i]

        retrieved_types = [
            metadata.iloc[int(idx)]["article_type"]
            for idx in retrieved_indices
        ]

        relevant = [
            int(article_type == row.articleType)
            for article_type in retrieved_types
        ]

        precision_at_1 = relevant[0]
        precision_at_5 = sum(relevant) / TOP_K
        hit_at_5 = int(any(relevant))

        reciprocal_rank = 0.0

        for rank, is_relevant in enumerate(
            relevant,
            start=1,
        ):
            if is_relevant:
                reciprocal_rank = 1.0 / rank
                break

        results.append({
            "product_id": int(row.id),
            "article_type": row.articleType,
            "gallery_type_count": int(
                type_counts.get(row.articleType, 0)
            ),
            "precision_at_1": precision_at_1,
            "precision_at_5": precision_at_5,
            "hit_at_5": hit_at_5,
            "reciprocal_rank_at_5": reciprocal_rank,
            "retrieved_ids": "|".join(
                str(metadata.iloc[int(idx)]["id"])
                for idx in retrieved_indices
            ),
            "retrieved_types": "|".join(
                retrieved_types
            ),
        })

# 8. Save per-query results
results_df = pd.DataFrame(results)

assert len(results_df) == len(val_df)

results_df.to_csv(
    OUTPUT_PATH,
    index=False,
)

# 9. Print aggregate metrics
print(
    f"\nFULL {args.split.upper()} RETRIEVAL RESULTS"
)
print("---------------------------------")
print("Queries:", len(results_df))
print("Gallery:", index.ntotal)

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

print(
    f"Category Coverage: "
    f"{(results_df['gallery_type_count'] > 0).mean():.2%}"
)

print("\nSaved:", OUTPUT_PATH)
