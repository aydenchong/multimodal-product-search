import numpy as np
import pandas as pd

METADATA_PATH = "data/processed/clip_product_metadata.csv"
RESULTS_PATH = "experiments/retrieval_evaluation.csv"

TOP_K = 5
SEED = 42

metadata = pd.read_csv(METADATA_PATH)
results = pd.read_csv(RESULTS_PATH)

rng = np.random.default_rng(SEED)

# 1. Count indexed products by article type
type_counts = metadata["article_type"].value_counts()

print("Index size:", len(metadata))
print("Unique article types:", len(type_counts))

print("\nTop 15 Indexed Article Types:")
print(type_counts.head(15).to_string())

# 2. Check whether each query type exists in the index
results["indexed_type_count"] = (
    results["article_type"]
    .map(type_counts)
    .fillna(0)
    .astype(int)
)

covered = results["indexed_type_count"] > 0

print("\nQuery Category Coverage")
print(f"Covered queries: {covered.sum()}/{len(results)}")
print(f"Coverage rate: {covered.mean():.2%}")

# 3. Simulate random retrieval
random_precision_1 = []
random_precision_5 = []
random_hit_5 = []
random_rr_5 = []

all_types = metadata["article_type"].to_numpy()

for row in results.itertuples():
    indices = rng.choice(
        len(metadata),
        size=TOP_K,
        replace=False,
    )

    retrieved_types = all_types[indices]

    relevant = (
        retrieved_types == row.article_type
    ).astype(int)

    random_precision_1.append(relevant[0])
    random_precision_5.append(relevant.mean())
    random_hit_5.append(int(relevant.any()))

    positions = np.flatnonzero(relevant)

    reciprocal_rank = (
        1.0 / (positions[0] + 1)
        if len(positions) > 0
        else 0.0
    )

    random_rr_5.append(reciprocal_rank)

# 4. Compare CLIP with random retrieval
comparison = pd.DataFrame({
    "Metric": [
        "Precision@1",
        "Precision@5",
        "Hit@5",
        "MRR@5",
    ],
    "CLIP_FAISS": [
        results["precision_at_1"].mean(),
        results["precision_at_5"].mean(),
        results["hit_at_5"].mean(),
        results["reciprocal_rank_at_5"].mean(),
    ],
    "Random": [
        np.mean(random_precision_1),
        np.mean(random_precision_5),
        np.mean(random_hit_5),
        np.mean(random_rr_5),
    ],
})

print("\nCLIP vs Random Retrieval:")
print(
    comparison.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}",
    )
)

# 5. Save comparison
comparison.to_csv(
    "experiments/retrieval_baseline_comparison.csv",
    index=False,
)

print("\nSaved baseline comparison.")
