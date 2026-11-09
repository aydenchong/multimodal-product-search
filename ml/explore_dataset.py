from datasets import load_dataset
import pandas as pd

DATASET_NAME = "ashraq/fashion-product-images-small"

print("Loading dataset...")

dataset = load_dataset(DATASET_NAME, split="train")

print("\nDataset information:")
print(dataset)

print("\nAvailable columns:")
print(dataset.column_names)

print("\nFirst sample metadata:")
sample = dataset[0]

for key, value in sample.items():
    if key == "image":
        print(f"{key}: {type(value).__name__}")
    else:
        print(f"{key}: {str(value)[:200]}")

# Convert dataset metadata into a DataFrame
df = dataset.remove_columns("image").to_pandas()

print("\nDataset shape:")
print(df.shape)

print("\nMaster category distribution:")
print(df["masterCategory"].value_counts())

print("\nArticle type distribution (Top 10):")
print(df["articleType"].value_counts().head(10))

print("\nMissing values:")
print(df.isnull().sum())

print("\nDuplicate product IDs:")
print(df["id"].duplicated().sum())