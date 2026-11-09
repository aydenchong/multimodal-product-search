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

from dataset import FashionDataset
from model import create_resnet18
from transforms import eval_transform


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
val_dataset = FashionDataset(
    "data/processed/val.csv",
    transform=eval_transform,
)

val_loader = DataLoader(
    val_dataset,
    batch_size=8,
    shuffle=False,
    num_workers=0,
)

# 2. Rebuild model architecture
model = create_resnet18(num_classes=4)

# 3. Load trained parameters
checkpoint_path = "models/resnet18_baseline.pth"

state_dict = torch.load(
    checkpoint_path,
    map_location="cpu",
    weights_only=True,
)

model.load_state_dict(state_dict)
model = model.to(device)
model.eval()

# 4. Run predictions
all_labels = []
all_predictions = []

with torch.inference_mode():
    for images, labels in val_loader:
        images = images.to(device)

        outputs = model(images)
        predictions = outputs.argmax(dim=1)

        all_labels.extend(labels.tolist())
        all_predictions.extend(predictions.cpu().tolist())

# 5. Calculate metrics
accuracy = accuracy_score(
    all_labels,
    all_predictions,
)

macro_f1 = f1_score(
    all_labels,
    all_predictions,
    average="macro",
    zero_division=0,
)

print(f"Validation Accuracy: {accuracy:.4f}")
print(f"Validation Macro F1: {macro_f1:.4f}")

print("\nClassification Report:")
print(
    classification_report(
        all_labels,
        all_predictions,
        labels=[0, 1, 2, 3],
        target_names=CLASS_NAMES,
        zero_division=0,
    )
)

# 6. Generate confusion matrix
cm = confusion_matrix(
    all_labels,
    all_predictions,
    labels=[0, 1, 2, 3],
)

output_dir = Path("experiments")
output_dir.mkdir(parents=True, exist_ok=True)

plt.figure(figsize=(9, 7))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=CLASS_NAMES,
    yticklabels=CLASS_NAMES,
)

plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("ResNet18 Validation Confusion Matrix")
plt.tight_layout()

output_path = output_dir / "resnet18_confusion_matrix.png"

plt.savefig(output_path, dpi=200)
plt.close()

print(f"\nConfusion matrix saved to {output_path}")
