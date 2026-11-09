import torch

# Training data
x = torch.tensor(2.0)
target = torch.tensor(10.0)

# Learnable parameter
w = torch.nn.Parameter(torch.tensor(1.0))

# Optimizer
optimizer = torch.optim.SGD([w], lr=0.1)

for epoch in range(20):

    # 1. Forward pass
    prediction = w * x

    # 2. Calculate loss
    loss = (prediction - target) ** 2

    # 3. Clear old gradients
    optimizer.zero_grad()

    # 4. Backpropagation
    loss.backward()

    # 5. Update parameter
    optimizer.step()

    print(
        f"Epoch {epoch + 1:02d} | "
        f"Weight: {w.item():.4f} | "
        f"Loss: {loss.item():.6f}"
    )

print("\nFinal prediction:", (w * x).item())
