from pathlib import Path

import faiss
import numpy as np
import pandas as pd

DATA_DIR = Path("data/processed")

EMBEDDINGS_PATH = DATA_DIR / "clip_full_embeddings.npy"
METADATA_PATH = DATA_DIR / "clip_full_metadata.csv"
INDEX_PATH = DATA_DIR / "clip_full_faiss.index"

# 1. Load full embeddings and metadata
embeddings = np.load(EMBEDDINGS_PATH)
metadata = pd.read_csv(METADATA_PATH)

print("Embeddings shape:", embeddings.shape)
print("Metadata rows:", len(metadata))

# 2. Validate embeddings
assert embeddings.ndim == 2
assert len(embeddings) == len(metadata)
assert metadata["id"].is_unique
assert np.isfinite(embeddings).all()

embeddings = np.ascontiguousarray(
    embeddings,
    dtype=np.float32,
)

# 3. Ensure unit-length vectors
faiss.normalize_L2(embeddings)

# 4. Build exact similarity search index
dimension = embeddings.shape[1]

index = faiss.IndexFlatIP(dimension)
index.add(embeddings)

print("FAISS vectors:", index.ntotal)
print("Embedding dimension:", index.d)

# 5. Save index
faiss.write_index(
    index,
    str(INDEX_PATH),
)

print("Saved:", INDEX_PATH)

# 6. Verify saved index
loaded_index = faiss.read_index(str(INDEX_PATH))

assert loaded_index.ntotal == len(metadata)
assert loaded_index.d == dimension

# 7. Test search with the first product
query = embeddings[0:1]

scores, indices = loaded_index.search(
    query,
    6,
)

query_product = metadata.iloc[0]

print("\nQuery Product:")
print(query_product["name"])
print("Article Type:", query_product["article_type"])

print("\nTop Similar Products:")

rank = 1

for score, idx in zip(scores[0], indices[0]):
    if idx < 0:
        continue

    product = metadata.iloc[idx]

    # Exclude the query product itself
    if product["id"] == query_product["id"]:
        continue

    print(
        f"{rank}. {product['name']} "
        f"| Type: {product['article_type']} "
        f"| Similarity: {score:.4f}"
    )

    rank += 1

    if rank > 5:
        break
