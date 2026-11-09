from torch.utils.data import DataLoader
from dataset import FashionDataset
from transforms import train_transform

train_dataset = FashionDataset(
    "data/processed/train.csv",
    transform=train_transform,
)

train_loader = DataLoader(
    train_dataset,
    batch_size=8,
    shuffle=True,
    num_workers=0,
)

images, labels = next(iter(train_loader))

print("Dataset size:", len(train_dataset))
print("Image batch shape:", images.shape)
print("Label batch shape:", labels.shape)
print("Labels:", labels)
print("Image dtype:", images.dtype)
