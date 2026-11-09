import random

import pandas as pd
import torch

from datasets import load_dataset
from transformers import CLIPModel, CLIPProcessor


MODEL_NAME = "openai/clip-vit-base-patch32"
SEED = 42
NUM_SAMPLES = 100
NUM_NEGATIVES = 3

random.seed(SEED)
torch.manual_seed(SEED)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)

# 1. Load validation metadata
val_df = pd.read_csv("data/processed/val.csv")

# 2. Load original dataset for images and product names
dataset = load_dataset(
    "ashraq/fashion-product-images-small",
    split="train",
)

id_to_index = {
    int(product_id): index
    for index, product_id in enumerate(dataset["id"])
}

# 3. Build metadata table for validation products
records = []

for row in val_df.itertuples():
    product_id = int(row.id)
    sample = dataset[id_to_index[product_id]]

    name = sample["productDisplayName"]

    if not isinstance(name, str) or not name.strip():
        continue

    records.append({
        "id": product_id,
        "name": name.strip(),
        "article_type": sample["articleType"],
    })

metadata = pd.DataFrame(records)

# Remove ambiguous duplicate product names
metadata = metadata.drop_duplicates(
    subset=["name"],
    keep=False,
).reset_index(drop=True)

# 4. Choose samples that have enough same-type negatives
eligible_rows = []

for row in metadata.itertuples():
    candidates = metadata[
        (metadata["article_type"] == row.article_type)
        & (metadata["id"] != row.id)
    ]

    if len(candidates) >= NUM_NEGATIVES:
        eligible_rows.append(row.Index)

selected = metadata.loc[eligible_rows].sample(
    n=min(NUM_SAMPLES, len(eligible_rows)),
    random_state=SEED,
)

# 5. Load pretrained CLIP
processor = CLIPProcessor.from_pretrained(MODEL_NAME)
model = CLIPModel.from_pretrained(MODEL_NAME)

model = model.to(device)
model.eval()

correct = 0
results = []

# 6. Evaluate image-text matching
for count, row in enumerate(selected.itertuples(), start=1):
    sample = dataset[id_to_index[row.id]]
    image = sample["image"].convert("RGB")

    negatives = metadata[
        (metadata["article_type"] == row.article_type)
        & (metadata["id"] != row.id)
    ].sample(
        n=NUM_NEGATIVES,
        random_state=SEED + count,
    )

    texts = [row.name] + negatives["name"].tolist()

    inputs = processor(
        images=image,
        text=texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
    )

    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    with torch.inference_mode():
        outputs = model(**inputs)

    scores = outputs.logits_per_image[0]
    predicted_index = scores.argmax().item()

    is_correct = predicted_index == 0
    correct += int(is_correct)

    results.append({
        "product_id": row.id,
        "true_name": row.name,
        "predicted_name": texts[predicted_index],
        "correct": is_correct,
    })

    if count % 20 == 0:
        print(f"Evaluated {count}/{len(selected)} products")

# 7. Final metrics
accuracy = correct / len(selected)

print("\nCLIP Matching Evaluation")
print("Samples:", len(selected))
print("Correct:", correct)
print(f"Top-1 Accuracy: {accuracy:.2%}")

# 8. Save results
from pathlib import Path

output_dir = Path("experiments")
output_dir.mkdir(parents=True, exist_ok=True)

pd.DataFrame(results).to_csv(
    output_dir / "clip_matching_results.csv",
    index=False,
)

print("Saved: experiments/clip_matching_results.csv")
