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

# 1. Parse command-line arguments
parser = argparse.ArgumentParser(
    description="Evaluate ResNet18 fashion classification"
)

parser.add_argument(
    "--split",
    choices=["val", "test"],
    default="val",
    help="Dataset split to evaluate",
)

args = parser.parse_args()

# 2. Configuration
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

dataset_path = Path("data/processed") / f"{args.split}.csv"

output_dir = Path("experiments")
output_dir.mkdir(parents=True, exist_ok=True)

print("Device:", device)
print("Dataset:", dataset_path)

# 3. Load evaluation dataset
eval_dataset = FashionDataset(
    str(dataset_path),
    transform=eval_transform,
)

eval_loader = DataLoader(
    eval_dataset,
    batch_size=8,
    shuffle=False,
    num_workers=0,
)

print("Evaluation images:", len(eval_dataset))

# 4. Rebuild model architecture
model = create_resnet18(num_classes=4)

# 5. Load trained parameters
checkpoint_path = "models/resnet18_baseline.pth"

state_dict = torch.load(
    checkpoint_path,
    map_location="cpu",
    weights_only=True,
)

model.load_state_dict(state_dict)
model = model.to(device)
model.eval()

# 6. Run predictions
all_labels = []
all_predictions = []

with torch.inference_mode():
    for images, labels in eval_loader:
        images = images.to(device)

        outputs = model(images)
        predictions = outputs.argmax(dim=1)

        all_labels.extend(labels.tolist())
        all_predictions.extend(
            predictions.cpu().tolist()
        )

# 7. Calculate metrics
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

report = classification_report(
    all_labels,
    all_predictions,
    labels=[0, 1, 2, 3],
    target_names=CLASS_NAMES,
    zero_division=0,
)

print(f"\n{args.split.upper()} CLASSIFICATION RESULTS")
print("---------------------------------")
print(f"Accuracy: {accuracy:.4%}")
print(f"Macro F1: {macro_f1:.4%}")

print("\nClassification Report:")
print(report)

# 8. Save classification report
report_path = (
    output_dir / f"resnet18_{args.split}_report.txt"
)

with open(report_path, "w", encoding="utf-8") as file:
    file.write(
        f"ResNet18 {args.split.upper()} Evaluation\n"
    )
    file.write(f"Images: {len(eval_dataset)}\n")
    file.write(f"Accuracy: {accuracy:.4%}\n")
    file.write(f"Macro F1: {macro_f1:.4%}\n\n")
    file.write(report)

# 9. Generate confusion matrix
cm = confusion_matrix(
    all_labels,
    all_predictions,
    labels=[0, 1, 2, 3],
)

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
plt.title(
    f"ResNet18 {args.split.upper()} Confusion Matrix"
)

plt.tight_layout()

matrix_path = (
    output_dir / f"resnet18_{args.split}_confusion_matrix.png"
)

plt.savefig(matrix_path, dpi=200)
plt.close()

print("\nSaved:")
print(report_path)
print(matrix_path)
