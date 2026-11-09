from collections import defaultdict
from hashlib import sha256
from io import BytesIO
import pandas as pd
from pathlib import Path
from datasets import load_dataset

DATASET_NAME = "ashraq/fashion-product-images-small"

dataset = load_dataset(DATASET_NAME, split="train")

# Group records by image hash
hash_groups = defaultdict(list)

for index, sample in enumerate(dataset):
    image = sample["image"].convert("RGB")

    buffer = BytesIO()
    image.save(buffer, format="PNG")

    image_hash = sha256(buffer.getvalue()).hexdigest()

    hash_groups[image_hash].append({
        "id": sample["id"],
        "masterCategory": sample["masterCategory"],
        "articleType": sample["articleType"],
    })

    if (index + 1) % 5000 == 0:
        print(f"Processed {index + 1} images")

duplicate_groups = [
    records
    for records in hash_groups.values()
    if len(records) > 1
]

master_conflicts = []
article_conflicts = []

for records in duplicate_groups:
    master_labels = {
        record["masterCategory"] for record in records
    }

    article_labels = {
        record["articleType"] for record in records
    }

    if len(master_labels) > 1:
        master_conflicts.append(records)

    if len(article_labels) > 1:
        article_conflicts.append(records)

print("\nDuplicate groups:", len(duplicate_groups))
print("Master category conflicts:", len(master_conflicts))
print("Article type conflicts:", len(article_conflicts))

print("\nExample article type conflicts:")

for group in article_conflicts[:5]:
    print(group)

# Step 1: Keep only the four main categories
VALID_CATEGORIES = {
    "Apparel",
    "Accessories",
    "Footwear",
    "Personal Care",
}

# Step 2: Identify hashes with conflicting master labels
conflicting_hashes = {
    image_hash
    for image_hash, records in hash_groups.items()
    if len({r["masterCategory"] for r in records}) > 1
}

# Step 3: Select one record per non-conflicting image
clean_records = []

for image_hash, records in hash_groups.items():
    if image_hash in conflicting_hashes:
        continue

    record = records[0]

    if record["masterCategory"] not in VALID_CATEGORIES:
        continue

    clean_records.append({
        **record,
        "image_hash": image_hash,
    })

# Step 4: Save metadata
output_dir = Path("data/processed")
output_dir.mkdir(parents=True, exist_ok=True)

df = pd.DataFrame(clean_records)
df.to_csv(output_dir / "clean_metadata.csv", index=False)

print("\nClean dataset size:", len(df))
print("\nClean category distribution:")
print(df["masterCategory"].value_counts())

print("\nSaved to:", output_dir / "clean_metadata.csv")
