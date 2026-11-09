from pathlib import Path

import faiss
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import textwrap
from datasets import load_dataset
from transformers import CLIPModel, CLIPProcessor

MODEL_NAME = "openai/clip-vit-base-patch32"
TOP_K = 5

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)

# 1. Load saved FAISS index and metadata
index = faiss.read_index(
    "data/processed/clip_faiss.index"
)

metadata = pd.read_csv(
    "data/processed/clip_product_metadata.csv"
)

assert index.ntotal == len(metadata)

# 2. Load a validation image
val_df = pd.read_csv(
    "data/processed/val.csv"
)

dataset = load_dataset(
    "ashraq/fashion-product-images-small",
    split="train",
)

id_to_index = {
    int(product_id): i
    for i, product_id in enumerate(dataset["id"])
}

query_id = int(val_df.iloc[0]["id"])
query_sample = dataset[id_to_index[query_id]]

query_image = query_sample["image"].convert("RGB")

print("\nQuery Product:")
print("ID:", query_id)
print("Name:", query_sample["productDisplayName"])
print("Category:", query_sample["masterCategory"])
print("Article Type:", query_sample["articleType"])

# 3. Load CLIP
processor = CLIPProcessor.from_pretrained(MODEL_NAME)
model = CLIPModel.from_pretrained(MODEL_NAME)

model = model.to(device)
model.eval()

# 4. Extract query embedding
inputs = processor(
    images=query_image,
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

query_vector = features.cpu().numpy().astype("float32")

# 5. Search FAISS
scores, indices = index.search(
    query_vector,
    TOP_K,
)

print("\nTop Similar Products:")

for rank, (score, result_index) in enumerate(
    zip(scores[0], indices[0]),
    start=1,
):
    if result_index < 0:
        continue

    product = metadata.iloc[result_index]

    print(
        f"{rank}. {product['name']} "
        f"| Similarity: {score:.4f}"
    )

# 6. Save query information
output_dir = Path("experiments")
output_dir.mkdir(parents=True, exist_ok=True)

# 7. Visualize query and retrieved products
fig, axes = plt.subplots(
    1,
    TOP_K + 1,
    figsize=(20, 5),
)

# Query image
axes[0].imshow(query_image)
axes[0].set_title(
    "QUERY\n" +
    textwrap.fill(
        query_sample["productDisplayName"],
        width=22,
    ),
    fontsize=10,
)
axes[0].axis("off")

# Retrieved images
for rank, (score, result_index) in enumerate(
    zip(scores[0], indices[0]),
    start=1,
):
    if result_index < 0:
        axes[rank].axis("off")
        continue

    product = metadata.iloc[result_index]
    product_id = int(product["id"])

    retrieved_sample = dataset[
        id_to_index[product_id]
    ]

    retrieved_image = retrieved_sample["image"].convert("RGB")

    axes[rank].imshow(retrieved_image)

    title = (
        f"Rank {rank} | {score:.3f}\n"
        + textwrap.fill(
            str(product["name"]),
            width=22,
        )
    )

    axes[rank].set_title(title, fontsize=9)
    axes[rank].axis("off")

plt.tight_layout()

output_path = (
    output_dir / "clip_faiss_retrieval.png"
)

plt.savefig(
    output_path,
    dpi=180,
    bbox_inches="tight",
)

plt.close()

print("Saved visualization:", output_path)

print("\nSearch completed.")
