import argparse
from pathlib import Path

import torch
import matplotlib.pyplot as plt
import seaborn as sns

from torch.utils.data import DataLoader
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score,
)

from ml.core.dataset import FashionDataset
from ml.core.model import create_resnet18
from ml.core.transforms import eval_transform


CLASS_NAMES = [
    "Apparel",
    "Accessories",
    "Footwear",
    "Personal Care",
]

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# 1. Load validation dataset
dataset = FashionDataset(
    "data/processed/val.csv",
    transform=eval_transform,
)

loader = DataLoader(
    dataset,
    batch_size=8,
    shuffle=False,
    num_workers=0,
)

# 2. Load trained model
model = create_resnet18(num_classes=4)

state_dict = torch.load(
    "models/resnet18_baseline.pth",
    map_location="cpu",
    weights_only=True,
)

model.load_state_dict(state_dict)
model = model.to(device)
model.eval()

# 3. Collect incorrect predictions
errors = []
sample_offset = 0

with torch.inference_mode():
    for images, labels in loader:
        outputs = model(images.to(device))
        probabilities = torch.softmax(outputs, dim=1)

        confidence, predictions = probabilities.max(dim=1)

        for i in range(len(labels)):
            true_label = labels[i].item()
            predicted_label = predictions[i].item()

            if true_label != predicted_label:
                errors.append({
                    "index": sample_offset + i,
                    "true": true_label,
                    "predicted": predicted_label,
                    "confidence": confidence[i].item(),
                })

        sample_offset += len(labels)

print("Total errors:", len(errors))

# 4. Focus on Accessories / Apparel confusion
selected_errors = [
    error for error in errors
    if (
        error["true"] == 1
        and error["predicted"] == 0
    ) or (
        error["true"] == 0
        and error["predicted"] == 1
    )
]

print("Accessories/Apparel errors:", len(selected_errors))

# Show the most confident incorrect predictions
selected_errors.sort(
    key=lambda error: error["confidence"],
    reverse=True,
)

selected_errors = selected_errors[:12]

# 5. Plot original images
fig, axes = plt.subplots(3, 4, figsize=(14, 12))

for ax, error in zip(axes.flat, selected_errors):
    row = dataset.df.iloc[error["index"]]

    product_id = int(row["id"])
    original_index = dataset.id_to_index[product_id]

    image = dataset.dataset[original_index]["image"]
    image = image.convert("RGB")

    ax.imshow(image)

    true_name = CLASS_NAMES[error["true"]]
    predicted_name = CLASS_NAMES[error["predicted"]]

    ax.set_title(
        f"True: {true_name}\n"
        f"Pred: {predicted_name}\n"
        f"Confidence: {error['confidence']:.1%}",
        fontsize=10,
    )

    ax.axis("off")

for ax in axes.flat[len(selected_errors):]:
    ax.axis("off")

plt.suptitle(
    "ResNet18 — Apparel / Accessories Errors",
    fontsize=16,
)

plt.tight_layout()

output_dir = Path("experiments")
output_dir.mkdir(parents=True, exist_ok=True)

output_path = output_dir / "resnet18_error_analysis.png"

plt.savefig(output_path, dpi=180)
plt.close()

print("Saved:", output_path)
