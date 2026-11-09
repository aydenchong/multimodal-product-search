from pathlib import Path

import pandas as pd

RESULTS_PATH = "experiments/retrieval_evaluation.csv"
METADATA_PATH = "data/processed/clip_product_metadata.csv"

results = pd.read_csv(RESULTS_PATH)
metadata = pd.read_csv(METADATA_PATH)

# 1. Count indexed products per article type
index_counts = metadata["article_type"].value_counts()

results["index_count"] = (
    results["article_type"]
    .map(index_counts)
    .fillna(0)
    .astype(int)
)

# 2. Aggregate retrieval metrics by article type
summary = (
    results.groupby("article_type")
    .agg(
        queries=("product_id", "count"),
        precision_at_1=("precision_at_1", "mean"),
        precision_at_5=("precision_at_5", "mean"),
        hit_at_5=("hit_at_5", "mean"),
        mrr_at_5=("reciprocal_rank_at_5", "mean"),
        index_count=("index_count", "first"),
    )
    .reset_index()
)

# 3. Focus on categories with enough queries
MIN_QUERIES = 5

eligible = summary[
    summary["queries"] >= MIN_QUERIES
].copy()

best = eligible.sort_values(
    "precision_at_5",
    ascending=False,
)

worst = eligible.sort_values(
    "precision_at_5",
    ascending=True,
)

columns = [
    "article_type",
    "queries",
    "index_count",
    "precision_at_1",
    "precision_at_5",
    "hit_at_5",
]

print("\nBest Performing Categories:")
print(best[columns].head(10).to_string(index=False))

print("\nWorst Performing Categories:")
print(worst[columns].head(10).to_string(index=False))

# 4. Analyze categories missing from the index
missing = results[
    results["index_count"] == 0
]

print("\nQueries with No Matching Indexed Type:")
print(
    missing[
        ["product_id", "product_name", "article_type"]
    ].to_string(index=False)
)

# 5. Save full category analysis
output_dir = Path("experiments")
output_dir.mkdir(parents=True, exist_ok=True)

summary.to_csv(
    output_dir / "retrieval_category_analysis.csv",
    index=False,
)

print("\nSaved: experiments/retrieval_category_analysis.csv")
