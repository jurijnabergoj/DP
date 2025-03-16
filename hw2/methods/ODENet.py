import torch
import torch.nn as nn
import torch.optim as optim


class ODENet(nn.Module):
    def __init__(self, hidden_dim=32):
        super(ODENet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        return self.net(x)


def train_ode_net(P, Q, n, model, x_range, initial_condition, num_steps=100, num_epochs=5000, learning_rate=0.001):
    """
    Trains the ODENet model to approximate the solution of a differential equation.

    Parameters:
    P: Function P(x) in the differential equation.
    Q: Function Q(x) in the differential equation.
    n: Exponent applied to y.
    model: Neural network model.
    x_range: Range of x values (start, end).
    initial_condition: Initial (x0, y0) values.
    num_steps: Number of x values for training. Defaults to 100.
    num_epochs: Number of training epochs. Defaults to 3000.
    learning_rate: Learning rate for optimization. Defaults to 0.001.

    Returns:
    ODENet: Trained neural network model.
    """
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    x0, y0 = initial_condition

    x = torch.linspace(x_range[0], x_range[1], num_steps + 1, requires_grad=True).view(-1, 1)

    # Initial point for the boundary condition
    x_initial = torch.tensor([x0], requires_grad=True).view(-1, 1)
    y_initial = torch.tensor([y0], dtype=torch.float32).view(-1, 1)

    for epoch in range(num_epochs):
        optimizer.zero_grad()

        # Forward pass
        y_pred = model(x)

        # Compute gradients
        y_pred_x = torch.autograd.grad(
            y_pred, x,
            grad_outputs=torch.ones_like(y_pred),
            create_graph=True
        )[0]

        P_tensor = torch.tensor(P(x.detach().numpy()), dtype=torch.float32).view(-1, 1)
        Q_tensor = torch.tensor(Q(x.detach().numpy()), dtype=torch.float32).view(-1, 1)

        residual = y_pred_x + P_tensor * y_pred - Q_tensor * y_pred ** n

        ode_loss = torch.mean(residual ** 2)
        bc_loss = torch.mean((model(x_initial) - y_initial) ** 2)

        loss = ode_loss + 10 * bc_loss

        loss.backward()
        optimizer.step()

        if (epoch + 1) % 500 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

    return model
