from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from datasets import load_dataset

# Load CLIP matching results
df = pd.read_csv(
    "experiments/clip_matching_results.csv"
)

errors = df[df["correct"] == False].copy()

print("Total errors:", len(errors))

# Load original product images
dataset = load_dataset(
    "ashraq/fashion-product-images-small",
    split="train",
)

id_to_index = {
    int(product_id): index
    for index, product_id in enumerate(dataset["id"])
}

# Display up to 12 incorrect matches
selected = errors.head(12)

fig, axes = plt.subplots(
    3,
    4,
    figsize=(18, 13),
)

for ax, row in zip(axes.flat, selected.itertuples()):
    sample = dataset[
        id_to_index[int(row.product_id)]
    ]

    image = sample["image"].convert("RGB")

    ax.imshow(image)

    title = (
        f"True: {row.true_name}\n"
        f"Pred: {row.predicted_name}"
    )

    ax.set_title(title, fontsize=8)
    ax.axis("off")

for ax in axes.flat[len(selected):]:
    ax.axis("off")

plt.tight_layout()

output_dir = Path("experiments")
output_dir.mkdir(parents=True, exist_ok=True)

output_path = output_dir / "clip_error_visualization.png"

plt.savefig(
    output_path,
    dpi=180,
    bbox_inches="tight",
)

plt.close()

print("Saved:", output_path)
