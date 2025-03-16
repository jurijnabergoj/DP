import os
import numpy as np
import torch
from matplotlib import pyplot as plt

from methods.laplaceNet import LaplaceNet, train_laplace_net
from methods.laplace_finite_diff import laplace_finite_difference

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


# Convert Laplace solution to velocity field
def simulate_potential_flow(u, nx, ny):
    """
    Convert Laplace solution to velocity field.

    Parameters:
    u: Solution to Laplace equation (potential function)
    x, y: Number of grid points for each axis

    Returns:
    vx, vy: Velocity components
    """
    vx = np.zeros((nx, ny))
    vy = np.zeros((nx, ny))

    # v = -grad(u)
    for i in range(1, nx - 1):
        for j in range(1, ny - 1):
            vx[i, j] = -(u[i + 1, j] - u[i - 1, j]) / (2 * dx)
            vy[i, j] = -(u[i, j + 1] - u[i, j - 1]) / (2 * dy)

    # Copy boundary values
    vx[0, :] = vx[1, :]
    vx[-1, :] = vx[-2, :]
    vx[:, 0] = vx[:, 1]
    vx[:, -1] = vx[:, -2]

    vy[0, :] = vy[1, :]
    vy[-1, :] = vy[-2, :]
    vy[:, 0] = vy[:, 1]
    vy[:, -1] = vy[:, -2]

    return vx, vy


def plot_solution(u, x, y, title):
    plt.figure(figsize=(10, 8))

    # Contour plot
    plt.subplot(2, 2, 1)
    cont = plt.contourf(x, y, u.T, 50, cmap='viridis')
    plt.colorbar(cont)
    plt.title(f'{title} - Contour')
    plt.xlabel('x')
    plt.ylabel('y')

    # Surface plot
    ax = plt.subplot(2, 2, 2, projection='3d')
    X, Y = np.meshgrid(x, y)
    ax.plot_surface(X, Y, u.T, cmap='viridis', edgecolor='none')
    ax.set_title(f'{title} - Surface')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('u')

    # Compute velocity field
    vx, vy = simulate_potential_flow(u, nx, ny)

    # Stream plot
    plt.subplot(2, 2, 3)
    speed = np.sqrt(vx ** 2 + vy ** 2).T
    plt.streamplot(x, y, vx.T, vy.T, density=1.5, color=speed, cmap='viridis')
    plt.colorbar(label='Speed')
    plt.title(f'{title} - Streamlines')
    plt.xlabel('x')
    plt.ylabel('y')

    # Quiver plot
    plt.subplot(2, 2, 4)
    skip = 2
    plt.quiver(x[::skip], y[::skip], vx.T[::skip, ::skip], vy.T[::skip, ::skip])
    plt.title(f'{title} - Velocity Vectors')
    plt.xlabel('x')
    plt.ylabel('y')

    plt.tight_layout()
    plt.show()


def laplace_equation_test():
    def bc_top(x):
        return 1.0  # Constant potential at top

    def bc_bottom(x):
        return 1.0  # Constant potential at bottom

    def bc_left(y):
        return 1.0  # Constant potential on left
        # return y / domain_size  # Linear potential on left

    def bc_right(y):
        return 1.0  # Constant potential on right

    u_fd, x_fd, y_fd = laplace_finite_difference(nx, ny, domain_size, bc_top, bc_bottom, bc_left, bc_right)
    plot_solution(u_fd, x_fd, y_fd, "Laplace Finite Difference")

    print("Training Neural Network for Laplace equation...")
    model = LaplaceNet()
    model = train_laplace_net(model, (bc_top, bc_bottom, bc_left, bc_right))

    # Generate grid
    x_nn = np.linspace(0, domain_size, nx)
    y_nn = np.linspace(0, domain_size, ny)
    X_nn, Y_nn = np.meshgrid(x_nn, y_nn)
    xy_nn = np.column_stack((X_nn.flatten(), Y_nn.flatten()))

    # Run NN on grid
    with torch.no_grad():
        u_nn = model(torch.tensor(xy_nn, dtype=torch.float32)).numpy().reshape(ny, nx).T

    plot_solution(u_nn, x_nn, y_nn, "Neural Network Solution")


if __name__ == "__main__":
    nx, ny = 50, 50  # Number of grid points
    domain_size = 1.0  # Domain size [0, 1] x [0, 1]
    dx = domain_size / (nx - 1)
    dy = domain_size / (ny - 1)

    laplace_equation_test()
