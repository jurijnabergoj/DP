import numpy as np


def laplace_finite_difference(nx, ny, domain_size, bc_top, bc_bottom, bc_left, bc_right, max_iter=1000, tol=1e-6):
    """
    Solves 2D Laplace equation using finite difference method.

    Parameters:
    bc_top, bc_bottom, bc_left, bc_right: Functions defining boundary conditions
    max_iter: Maximum number of iterations
    tol: Convergence tolerance

    Returns:
    u: Solution array
    """
    x = np.linspace(0, domain_size, nx)
    y = np.linspace(0, domain_size, ny)

    # Initialize solution with boundary conditions
    u = np.zeros((nx, ny))

    # Set boundary conditions
    for i in range(nx):
        u[i, 0] = bc_bottom(x[i])
        u[i, -1] = bc_top(x[i])

    for j in range(ny):
        u[0, j] = bc_left(y[j])
        u[-1, j] = bc_right(y[j])

    # Iterative Jacobi method
    for it in range(max_iter):
        u_old = u.copy()

        # Update interior points
        for i in range(1, nx - 1):
            for j in range(1, ny - 1):
                u[i, j] = 0.25 * (u_old[i + 1, j] + u_old[i - 1, j] + u_old[i, j + 1] + u_old[i, j - 1])

        # Check convergence
        diff = np.max(np.abs(u - u_old))
        if diff < tol:
            print(f"Converged after {it + 1} iterations")
            break

    return u, x, y
