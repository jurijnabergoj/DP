import numpy as np


def runge_kutta_4(Q, P, n, x0, y0, x_end, num_steps):
    """
    Solves the differential equation using the 4th-order Runge-Kutta method.

    Parameters:
    Q: Function Q(x) in the differential equation.
    P: Function P(x) in the differential equation.
    n: Exponent applied to y.
    x0: Initial x value.
    y0: Initial y value.
    x_end: The final x value.
    num_steps: Number of steps for the numerical integration.

    Returns:
    tuple: Arrays of x values and corresponding y values.
    """
    h = (x_end - x0) / num_steps
    x = np.linspace(x0, x_end, num_steps + 1)
    y = np.zeros(num_steps + 1)
    y[0] = y0

    for i in range(num_steps):
        k1 = h * (Q(x[i]) * y[i] ** n - P(x[i]) * y[i])
        k2 = h * (Q(x[i] + 0.5 * h) * (y[i] + 0.5 * k1) ** n - P(x[i] + 0.5 * h) * (y[i] + 0.5 * k1))
        k3 = h * (Q(x[i] + 0.5 * h) * (y[i] + 0.5 * k2) ** n - P(x[i] + 0.5 * h) * (y[i] + 0.5 * k2))
        k4 = h * (Q(x[i] + h) * (y[i] + k3) ** n - P(x[i] + h) * (y[i] + k3))

        y[i + 1] = y[i] + (k1 + 2 * k2 + 2 * k3 + k4) / 6

    return x, y
