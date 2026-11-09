import torch

# Create a simple tensor
x = torch.tensor([
    [1.0, 2.0],
    [3.0, 4.0]
])

print("Tensor:")
print(x)

print("\nShape:", x.shape)
print("Data type:", x.dtype)
print("Device:", x.device)

# Move tensor to GPU if available
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

x = x.to(device)

print("\nAfter moving:")
print("Device:", x.device)

# Tensor operations
y = x * 2

print("\nResult:")
print(y)
