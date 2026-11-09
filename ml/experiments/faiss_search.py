from pathlib import Path

import faiss
import numpy as np
import pandas as pd

EMBEDDINGS_PATH = Path(
    "data/processed/clip_image_embeddings.npy"
)

METADATA_PATH = Path(
    "data/processed/clip_product_metadata.csv"
)

INDEX_PATH = Path(
    "data/processed/clip_faiss.index"
)

TOP_K = 5

# 1. Load embeddings and metadata
embeddings = np.load(EMBEDDINGS_PATH).astype("float32")
metadata = pd.read_csv(METADATA_PATH)

print("Embedding shape:", embeddings.shape)
print("Metadata rows:", len(metadata))

# 2. Validate data
assert len(embeddings) == len(metadata), (
    "Embedding and metadata counts do not match"
)

dimension = embeddings.shape[1]

# 3. Build FAISS index
index = faiss.IndexFlatIP(dimension)
index.add(embeddings)

print("FAISS index size:", index.ntotal)

# 4. Save index
faiss.write_index(index, str(INDEX_PATH))

print("Saved index:", INDEX_PATH)

# 5. Select a query from the indexed products
query_index = 0
query_vector = embeddings[
    query_index:query_index + 1
]

print("\nQuery Product:")
print(metadata.iloc[query_index].to_string())

# 6. Search for nearest neighbors
scores, indices = index.search(
    query_vector,
    TOP_K + 1,
)

print("\nTop Similar Products:")

rank = 0

for score, result_index in zip(
    scores[0],
    indices[0],
):
    if result_index == query_index:
        continue

    if result_index < 0:
        continue

    rank += 1

    product = metadata.iloc[result_index]

    print(
        f"{rank}. {product['name']} "
        f"| Similarity: {score:.4f}"
    )

    if rank == TOP_K:
        break
