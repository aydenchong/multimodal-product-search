import pandas as pd
import torch

from torch.utils.data import Dataset
from torchvision import transforms
from datasets import load_dataset


class FashionDataset(Dataset):

    LABEL_MAP = {
        "Apparel": 0,
        "Accessories": 1,
        "Footwear": 2,
        "Personal Care": 3,
    }

    def __init__(self, csv_path, transform=None):
        self.df = pd.read_csv(csv_path)
        self.transform = transform
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

        self.dataset = load_dataset(
            "ashraq/fashion-product-images-small",
            split="train",
        )

        # Map product IDs to original dataset indices
        self.id_to_index = {
            int(product_id): index
            for index, product_id in enumerate(self.dataset["id"])
        }

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]

        product_id = int(row["id"])
        label_name = row["masterCategory"]

        original_index = self.id_to_index[product_id]

        image = self.dataset[original_index]["image"]
        image = image.convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        label = self.LABEL_MAP[label_name]

        return image, torch.tensor(label, dtype=torch.long)
