import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

MODEL_NAME = "openai/clip-vit-base-patch32"

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)

# Load pretrained CLIP
processor = CLIPProcessor.from_pretrained(MODEL_NAME)
model = CLIPModel.from_pretrained(MODEL_NAME)

model = model.to(device)
model.eval()

# Create a simple test image
image = Image.new(
    "RGB",
    (224, 224),
    color="red",
)

texts = [
    "A red square",
    "A blue square",
    "A pair of shoes",
]

# Convert image and text into model inputs
inputs = processor(
    text=texts,
    images=image,
    return_tensors="pt",
    padding=True,
)

inputs = {
    key: value.to(device)
    for key, value in inputs.items()
}

# Forward pass
with torch.inference_mode():
    outputs = model(**inputs)

# CLIP image-text similarity logits
logits = outputs.logits_per_image

print("Logits shape:", logits.shape)
print("Similarity logits:", logits.cpu())

# Relative probabilities among these three texts
probabilities = logits.softmax(dim=1)

for text, score in zip(
    texts,
    probabilities[0].cpu().tolist(),
):
    print(f"{text}: {score:.2%}")
