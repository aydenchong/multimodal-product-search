import faiss
import numpy as np
import pandas as pd
import torch

from PIL import Image
from transformers import CLIPModel, CLIPProcessor

from backend.app.config import (
    CLIP_MODEL_NAME,
    FAISS_INDEX_PATH,
    PRODUCT_METADATA_PATH,
)


class RetrievalService:
    def __init__(self):
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.processor = CLIPProcessor.from_pretrained(
            CLIP_MODEL_NAME
        )

        self.model = CLIPModel.from_pretrained(
            CLIP_MODEL_NAME
        )

        self.model.to(self.device)
        self.model.eval()

        self.index = faiss.read_index(
            str(FAISS_INDEX_PATH)
        )

        self.metadata = pd.read_csv(
            PRODUCT_METADATA_PATH
        ).fillna("")

        if self.index.ntotal != len(self.metadata):
            raise ValueError(
                "FAISS index and metadata size mismatch"
            )

        if self.index.d != 512:
            raise ValueError(
                "Unexpected FAISS embedding dimension"
            )

    @torch.inference_mode()
    def search(
        self,
        image: Image.Image,
        top_k: int = 5,
    ) -> list[dict]:
        inputs = self.processor(
            images=image.convert("RGB"),
            return_tensors="pt",
        )

        pixel_values = inputs["pixel_values"].to(
            self.device
        )

        features = self.model.get_image_features(
            pixel_values=pixel_values
        )

        if not isinstance(features, torch.Tensor):
            features = features.pooler_output

        features = torch.nn.functional.normalize(
            features,
            p=2,
            dim=-1,
        )

        query_vector = np.ascontiguousarray(
            features.cpu().numpy(),
            dtype=np.float32,
        )

        scores, indices = self.index.search(
            query_vector,
            min(top_k, self.index.ntotal),
        )

        results = []

        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue

            product = self.metadata.iloc[int(idx)]

            results.append({
                "id": int(product["id"]),
                "name": str(product["name"]),
                "category": str(product["category"]),
                "article_type": str(product["article_type"]),
                "similarity": round(float(score), 4),
            })

        return results
