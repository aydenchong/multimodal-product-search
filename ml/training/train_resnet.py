from pathlib import Path

import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score

from ml.core.dataset import FashionDataset
from ml.core.model import create_resnet18
from ml.core.transforms import train_transform, eval_transform


# Configuration
BATCH_SIZE = 8
LEARNING_RATE = 0.001
EPOCHS = 1

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Training device:", device)

# Reproducibility
torch.manual_seed(42)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# Datasets
train_dataset = FashionDataset(
    "data/processed/train.csv",
    transform=train_transform,
)

val_dataset = FashionDataset(
    "data/processed/val.csv",
    transform=eval_transform,
)

# DataLoaders
train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
)

# Model
model = create_resnet18(num_classes=4)
model = model.to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.fc.parameters(),
    lr=LEARNING_RATE,
)

# Training
for epoch in range(EPOCHS):
    model.train()

    # Keep the frozen backbone BatchNorm statistics unchanged
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            module.eval()

    running_loss = 0.0
    total_samples = 0

    for batch_idx, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        total_samples += images.size(0)

        if (batch_idx + 1) % 200 == 0:
            print(
                f"Epoch {epoch + 1} | "
                f"Batch {batch_idx + 1}/{len(train_loader)} | "
                f"Loss: {loss.item():.4f}",
                flush=True,
            )

    train_loss = running_loss / total_samples

    # Validation
    model.eval()

    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)

            outputs = model(images)
            predictions = outputs.argmax(dim=1)

            all_predictions.extend(predictions.cpu().tolist())
            all_labels.extend(labels.tolist())

    val_accuracy = accuracy_score(
        all_labels,
        all_predictions,
    )

    val_macro_f1 = f1_score(
        all_labels,
        all_predictions,
        average="macro",
        zero_division=0,
    )

    print(f"\nEpoch {epoch + 1} Results")
    print(f"Train Loss: {train_loss:.4f}")
    print(f"Validation Accuracy: {val_accuracy:.4f}")
    print(f"Validation Macro F1: {val_macro_f1:.4f}")

# Save checkpoint
output_dir = Path("models")
output_dir.mkdir(parents=True, exist_ok=True)

torch.save(
    model.state_dict(),
    output_dir / "resnet18_baseline.pth",
)

print("\nModel saved to models/resnet18_baseline.pth")
