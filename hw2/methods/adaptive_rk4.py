import numpy as np


def adaptive_rk4(Q, P, n, x0, y0, x_end, tol=1e-6, h_min=1e-10, h_max=0.1):
    """
    Solves the differential equation dy/dx = Q(x)*y^n - P(x)*y using an adaptive Runge-Kutta 4th order method.

    Parameters:
    Q: Function Q(x) in the differential equation.
    P: Function P(x) in the differential equation.
    n: Exponent applied to y.
    x0: Initial x value.
    y0: Initial y value.
    x_end: The final x value.
    tol: Tolerance for error control. Defaults to 1e-6.
    h_min: Minimum step size. Defaults to 1e-10.
    h_max: Maximum step size. Defaults to 0.1.

    Returns:
    tuple: Arrays of x values and corresponding y values.
    """
    x = [x0]
    y = [y0]
    h = h_max

    while x[-1] < x_end:
        if x[-1] + h > x_end:
            h = x_end - x[-1]

        k1 = h * (Q(x[-1]) * y[-1] ** n - P(x[-1]) * y[-1])
        k2 = h * (Q(x[-1] + 0.5 * h) * (y[-1] + 0.5 * k1) ** n - P(x[-1] + 0.5 * h) * (y[-1] + 0.5 * k1))
        k3 = h * (Q(x[-1] + 0.5 * h) * (y[-1] + 0.5 * k2) ** n - P(x[-1] + 0.5 * h) * (y[-1] + 0.5 * k2))
        k4 = h * (Q(x[-1] + h) * (y[-1] + k3) ** n - P(x[-1] + h) * (y[-1] + k3))

        y_new_full = y[-1] + (k1 + 2 * k2 + 2 * k3 + k4) / 6

        h_half = h / 2
        k1_half = h_half * (Q(x[-1]) * y[-1] ** n - P(x[-1]) * y[-1])
        k2_half = h_half * (Q(x[-1] + 0.5 * h_half) * (y[-1] + 0.5 * k1_half) ** n - P(x[-1] + 0.5 * h_half) * (
                y[-1] + 0.5 * k1_half))
        k3_half = h_half * (Q(x[-1] + 0.5 * h_half) * (y[-1] + 0.5 * k2_half) ** n - P(x[-1] + 0.5 * h_half) * (
                y[-1] + 0.5 * k2_half))
        k4_half = h_half * (Q(x[-1] + h_half) * (y[-1] + k3_half) ** n - P(x[-1] + h_half) * (y[-1] + k3_half))

        y_mid = y[-1] + (k1_half + 2 * k2_half + 2 * k3_half + k4_half) / 6

        k1_half = h_half * (Q(x[-1] + h_half) * y_mid ** n - P(x[-1] + h_half) * y_mid)
        k2_half = h_half * (Q(x[-1] + h_half + 0.5 * h_half) * (y_mid + 0.5 * k1_half) ** n - P(
            x[-1] + h_half + 0.5 * h_half) * (y_mid + 0.5 * k1_half))
        k3_half = h_half * (Q(x[-1] + h_half + 0.5 * h_half) * (y_mid + 0.5 * k2_half) ** n - P(
            x[-1] + h_half + 0.5 * h_half) * (y_mid + 0.5 * k2_half))
        k4_half = h_half * (Q(x[-1] + h) * (y_mid + k3_half) ** n - P(x[-1] + h) * (y_mid + k3_half))

        y_new_half = y_mid + (k1_half + 2 * k2_half + 2 * k3_half + k4_half) / 6

        error = abs(y_new_full - y_new_half)

        if error < tol:
            x.append(x[-1] + h)
            y.append(y_new_half)

            if error < tol / 10:
                h = min(h * 2, h_max)
        else:
            h = max(h / 2, h_min)
            if h == h_min:
                x.append(x[-1] + h)
                y.append(y_new_half)

    return np.array(x), np.array(y)
