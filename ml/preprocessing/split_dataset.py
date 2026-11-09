from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

RANDOM_SEED = 42

DATA_PATH = Path("data/processed/clean_metadata.csv")
OUTPUT_DIR = Path("data/processed")

df = pd.read_csv(DATA_PATH)

# First split: 70% train, 30% temporary
train_df, temp_df = train_test_split(
    df,
    test_size=0.30,
    stratify=df["masterCategory"],
    random_state=RANDOM_SEED,
)

# Second split: temporary data -> 50% validation, 50% test
val_df, test_df = train_test_split(
    temp_df,
    test_size=0.50,
    stratify=temp_df["masterCategory"],
    random_state=RANDOM_SEED,
)

# Save the three splits
train_df.to_csv(OUTPUT_DIR / "train.csv", index=False)
val_df.to_csv(OUTPUT_DIR / "val.csv", index=False)
test_df.to_csv(OUTPUT_DIR / "test.csv", index=False)

print("Train:", len(train_df))
print("Validation:", len(val_df))
print("Test:", len(test_df))

print("\nCategory distribution (%):")

for name, split_df in [
    ("Train", train_df),
    ("Validation", val_df),
    ("Test", test_df),
]:
    print(f"\n{name}")
    print(
        (
            split_df["masterCategory"]
            .value_counts(normalize=True)
            .mul(100)
            .round(2)
        )
    )

# Check for duplicate image hashes across splits
train_hashes = set(train_df["image_hash"])
val_hashes = set(val_df["image_hash"])
test_hashes = set(test_df["image_hash"])

assert train_hashes.isdisjoint(val_hashes)
assert train_hashes.isdisjoint(test_hashes)
assert val_hashes.isdisjoint(test_hashes)

# Check for duplicate product IDs across splits
assert set(train_df["id"]).isdisjoint(set(val_df["id"]))
assert set(train_df["id"]).isdisjoint(set(test_df["id"]))
assert set(val_df["id"]).isdisjoint(set(test_df["id"]))

print("\nSplit integrity checks passed!")
