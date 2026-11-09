import torch
from model import create_resnet18


device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

model = create_resnet18(num_classes=4)
model = model.to(device)

# Evaluation mode for this smoke test
model.eval()

# Simulate a batch of 8 RGB images
dummy_images = torch.randn(
    8, 3, 224, 224,
    device=device,
)

with torch.no_grad():
    outputs = model(dummy_images)

print("Device:", device)
print("Output shape:", outputs.shape)

total_params = sum(
    p.numel() for p in model.parameters()
)

trainable_params = sum(
    p.numel()
    for p in model.parameters()
    if p.requires_grad
)

print("Total parameters:", total_params)
print("Trainable parameters:", trainable_params)
