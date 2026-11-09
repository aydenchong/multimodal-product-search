import torch
from datasets import load_dataset
from transformers import CLIPModel, CLIPProcessor

MODEL_NAME = "openai/clip-vit-base-patch32"

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# Load pretrained CLIP
processor = CLIPProcessor.from_pretrained(MODEL_NAME)
model = CLIPModel.from_pretrained(MODEL_NAME)

model = model.to(device)
model.eval()

# Load fashion dataset
dataset = load_dataset(
    "ashraq/fashion-product-images-small",
    split="train",
)

# Select a real product
sample = dataset[0]

image = sample["image"].convert("RGB")

print("Product:", sample["productDisplayName"])
print("True Category:", sample["masterCategory"])
print("Article Type:", sample["articleType"])
print("Image Size:", image.size)

# Candidate descriptions
texts = [
    "A product photo of clothing",
    "A product photo of fashion accessories",
    "A product photo of footwear",
    "A product photo of personal care products",
]

# Preprocess image and text
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

# Inference
with torch.inference_mode():
    outputs = model(**inputs)

probabilities = outputs.logits_per_image.softmax(dim=1)[0]

# Print ranked matches
results = sorted(
    zip(texts, probabilities.cpu().tolist()),
    key=lambda item: item[1],
    reverse=True,
)

print("\nCLIP Matching Results:")

for description, score in results:
    print(f"{score:.2%} | {description}")
