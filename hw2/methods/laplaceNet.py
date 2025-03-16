import torch
from torch import nn, optim


class LaplaceNet(nn.Module):
    def __init__(self, hidden_dim=64):
        super(LaplaceNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        return self.net(x)


def train_laplace_net(model, bc_functions, domain_size=1.0, num_epochs=5000, learning_rate=0.001):
    """
    Train neural network to solve 2D Laplace equation

    Parameters:
    model: Neural network model
    bc_functions: Tuple of (bc_top, bc_bottom, bc_left, bc_right) boundary condition functions
    domain_size: Size of the square domain
    num_epochs: Number of training epochs
    learning_rate: Learning rate for optimizer

    Returns:
    model: Trained model
    """
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    bc_top, bc_bottom, bc_left, bc_right = bc_functions

    # Define number of interior and boundary points
    n_interior = 5000
    n_boundary = 1000

    for epoch in range(num_epochs):
        optimizer.zero_grad()

        # Sample interior points
        x_interior = torch.rand(n_interior, 2) * domain_size
        x_interior.requires_grad = True

        # Forward pass for interior points
        u_interior = model(x_interior)

        # Gradients for Laplacian
        grads = torch.autograd.grad(
            u_interior, x_interior,
            grad_outputs=torch.ones_like(u_interior),
            create_graph=True
        )[0]

        # Second derivatives in x
        u_xx = torch.autograd.grad(
            grads[:, 0], x_interior,
            grad_outputs=torch.ones_like(grads[:, 0]),
            create_graph=True
        )[0][:, 0]

        # Second derivatives in y
        u_yy = torch.autograd.grad(
            grads[:, 1], x_interior,
            grad_outputs=torch.ones_like(grads[:, 1]),
            create_graph=True
        )[0][:, 1]

        # PDE residual (Laplace equation: u_xx + u_yy = 0)
        pde_residual = u_xx + u_yy

        # PDE loss
        pde_loss = torch.mean(pde_residual ** 2)

        # Sample boundary points
        x_top = torch.cat([torch.rand(n_boundary // 4, 1) * domain_size,
                           torch.ones(n_boundary // 4, 1) * domain_size], dim=1)
        x_bottom = torch.cat([torch.rand(n_boundary // 4, 1) * domain_size,
                              torch.zeros(n_boundary // 4, 1)], dim=1)
        x_left = torch.cat([torch.zeros(n_boundary // 4, 1),
                            torch.rand(n_boundary // 4, 1) * domain_size], dim=1)
        x_right = torch.cat([torch.ones(n_boundary // 4, 1) * domain_size,
                             torch.rand(n_boundary // 4, 1) * domain_size], dim=1)

        # Compute boundary values
        u_top_true = torch.tensor([bc_top(x.item()) for x in x_top[:, 0]]).unsqueeze(1)
        u_bottom_true = torch.tensor([bc_bottom(x.item()) for x in x_bottom[:, 0]]).unsqueeze(1)
        u_left_true = torch.tensor([bc_left(y.item()) for y in x_left[:, 1]]).unsqueeze(1)
        u_right_true = torch.tensor([bc_right(y.item()) for y in x_right[:, 1]]).unsqueeze(1)

        # Forward pass for boundary points
        u_top_pred = model(x_top)
        u_bottom_pred = model(x_bottom)
        u_left_pred = model(x_left)
        u_right_pred = model(x_right)

        # Boundary loss
        bc_loss = (torch.mean((u_top_pred - u_top_true) ** 2) +
                   torch.mean((u_bottom_pred - u_bottom_true) ** 2) +
                   torch.mean((u_left_pred - u_left_true) ** 2) +
                   torch.mean((u_right_pred - u_right_true) ** 2))

        # Total loss
        loss = pde_loss + 10 * bc_loss  # Weight boundary conditions more heavily

        loss.backward()
        optimizer.step()

        if (epoch + 1) % 500 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], PDE Loss: {pde_loss.item():.6f}, BC Loss: {bc_loss.item():.6f}')

    return model
