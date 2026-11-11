import torch
from PIL import Image

from backend.app.config import CLASS_NAMES, MODEL_PATH
from ml.core.model import create_resnet18
from ml.core.transforms import eval_transform


class ClassificationService:
    def __init__(self):
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.model = create_resnet18(num_classes=4)

        state_dict = torch.load(
            MODEL_PATH,
            map_location="cpu",
            weights_only=True,
        )

        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

    @torch.inference_mode()
    def predict(self, image: Image.Image) -> dict:
        tensor = eval_transform(image.convert("RGB"))
        tensor = tensor.unsqueeze(0).to(self.device)

        logits = self.model(tensor)
        probabilities = torch.softmax(logits, dim=1)

        confidence, predicted_index = probabilities.max(dim=1)

        return {
            "category": CLASS_NAMES[predicted_index.item()],
            "confidence": round(confidence.item(), 4),
        }
