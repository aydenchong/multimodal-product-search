from pathlib import Path
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

# 1. Load validation metadata and original images
val_df = pd.read_csv("data/processed/val.csv")

dataset = load_dataset(
    "ashraq/fashion-product-images-small",
    split="train",
)

id_to_index = {
    int(product_id): index
    for index, product_id in enumerate(dataset["id"])
}

# 2. Build metadata
records = []

for row in val_df.itertuples():
    sample = dataset[id_to_index[int(row.id)]]

    name = sample["productDisplayName"]

    if not isinstance(name, str) or not name.strip():
        continue

    records.append({
        "id": int(row.id),
        "name": name.strip(),
        "article_type": sample["articleType"],
        "gender": sample["gender"],
        "colour": sample["baseColour"],
    })

metadata = pd.DataFrame(records)

# Match v1: exclude duplicated product names
metadata = metadata.drop_duplicates(
    subset=["name"],
    keep=False,
).reset_index(drop=True)

# 3. Build standardized descriptions
def visual_description(row):
    gender = str(row["gender"]).strip()
    colour = str(row["colour"]).strip()
    article_type = str(row["article_type"]).strip()

    return (
        f"A product photo of {colour} "
        f"{article_type} for {gender}"
    )

metadata["visual_text"] = metadata.apply(
    visual_description,
    axis=1,
)

# 4. Select eligible products
eligible_rows = []

for row in metadata.itertuples():
    candidates = metadata[
        (metadata["article_type"] == row.article_type)
        & (metadata["id"] != row.id)
        & (metadata["visual_text"] != row.visual_text)
    ]

    if len(candidates) >= NUM_NEGATIVES:
        eligible_rows.append(row.Index)

selected = metadata.loc[eligible_rows].sample(
    n=min(NUM_SAMPLES, len(eligible_rows)),
    random_state=SEED,
)

# 5. Load CLIP
processor = CLIPProcessor.from_pretrained(MODEL_NAME)
model = CLIPModel.from_pretrained(MODEL_NAME)

model = model.to(device)
model.eval()

results = []

# 6. Evaluate both prompt types
for count, row in enumerate(selected.itertuples(), start=1):
    sample = dataset[id_to_index[row.id]]
    image = sample["image"].convert("RGB")

    negatives = metadata[
        (metadata["article_type"] == row.article_type)
        & (metadata["id"] != row.id)
        & (metadata["visual_text"] != row.visual_text)
    ].sample(
        n=NUM_NEGATIVES,
        random_state=SEED + count,
    )

    candidates = pd.concat([
        metadata[metadata["id"] == row.id],
        negatives,
    ])

    # Randomize candidate order
    candidates = candidates.sample(
        frac=1,
        random_state=SEED + count,
    ).reset_index(drop=True)

    true_index = candidates.index[
        candidates["id"] == row.id
    ][0]

    experiment_scores = {}

    for experiment, text_column in [
        ("original", "name"),
        ("visual", "visual_text"),
    ]:
        texts = candidates[text_column].tolist()

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

        experiment_scores[experiment] = {
            "correct": predicted_index == true_index,
            "predicted": texts[predicted_index],
            "scores": scores.cpu().tolist(),
        }

    results.append({
        "product_id": row.id,
        "true_name": row.name,
        "original_correct": experiment_scores[
            "original"
        ]["correct"],
        "visual_correct": experiment_scores[
            "visual"
        ]["correct"],
        "original_prediction": experiment_scores[
            "original"
        ]["predicted"],
        "visual_prediction": experiment_scores[
            "visual"
        ]["predicted"],
    })

    if count % 20 == 0:
        print(f"Evaluated {count}/{len(selected)}")

# 7. Calculate metrics
results_df = pd.DataFrame(results)

original_accuracy = results_df[
    "original_correct"
].mean()

visual_accuracy = results_df[
    "visual_correct"
].mean()

print("\nCLIP Evaluation v2")
print("Samples:", len(results_df))
print(f"Original Name Accuracy: {original_accuracy:.2%}")
print(f"Visual Description Accuracy: {visual_accuracy:.2%}")

# 8. Save results
output_dir = Path("experiments")
output_dir.mkdir(parents=True, exist_ok=True)

results_df.to_csv(
    output_dir / "clip_evaluation_v2.csv",
    index=False,
)

print("Saved: experiments/clip_evaluation_v2.csv")
