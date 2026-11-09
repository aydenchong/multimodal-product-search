import matplotlib.pyplot as plt
import torch
from dataset import FashionDataset
from transforms import train_transform, eval_transform


# Load the same training split with different transforms
original_dataset = FashionDataset(
    "data/processed/train.csv",
    transform=eval_transform,
)

augmented_dataset = FashionDataset(
    "data/processed/train.csv",
    transform=train_transform,
)

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def denormalize(image):
    image = image.cpu() * STD + MEAN
    return image.clamp(0, 1).permute(1, 2, 0).numpy()


# Choose one product
sample_index = 0
row = original_dataset.df.iloc[sample_index]
product_id = int(row["id"])

original_index = original_dataset.id_to_index[product_id]
raw_image = original_dataset.dataset[original_index]["image"]

print("Original image size:", raw_image.size)
print("Image mode:", raw_image.mode)

original_image, label = original_dataset[sample_index]

fig, axes = plt.subplots(1, 5, figsize=(15, 3))

axes[0].imshow(denormalize(original_image))
axes[0].set_title("Original")

for i in range(1, 5):
    augmented_image, _ = augmented_dataset[sample_index]

    axes[i].imshow(denormalize(augmented_image))
    axes[i].set_title(f"Augmented {i}")

for ax in axes:
    ax.axis("off")

plt.tight_layout()
plt.savefig(
    "data/processed/augmentation_preview.png",
    dpi=150,
)
plt.show()
