import torch

# Create a tensor that tracks gradients
x = torch.tensor(3.0, requires_grad=True)

# Forward pass
y = x**2 + 2*x + 1

print("x:", x.item())
print("y:", y.item())

# Backward pass
y.backward()

print("Gradient:", x.grad.item())
