import numpy as np
import torch
import os
from matplotlib import pyplot as plt
from scipy.integrate import solve_ivp

from methods.ODENet import ODENet, train_ode_net
from methods.adaptive_rk4 import adaptive_rk4
from methods.euler import euler_method
from methods.rk4 import runge_kutta_4

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


# Bernoulli equation: dy/dx + y = x*y^2
# Parameters: P(x) = 1, Q(x) = x, n = 2
# Initial condition: y(0) = 1
# Exact solution: x^2 -2x + 2 - e^(-x)
def P(x):
    return 1.0


def Q(x):
    return x ** 2


def exact_solution(x):
    return (x ** 2 - 2 * x + 2) - np.exp(-x)


def bernoulli_rhs(x, y):
    return Q(x) * y ** n - P(x) * y


if __name__ == "__main__":
    n = 0
    x0 = 0.0
    y0 = 1.0
    x_span = [0.0, 2.0]
    n_steps = 100
    x_vals = np.linspace(x_span[0], x_span[1], n_steps + 1)

    x_euler, y_euler = euler_method(Q=Q, P=P, n=n, x0=x0, y0=y0, x_end=x_span[1], num_steps=n_steps)
    x_rk4, y_rk4 = runge_kutta_4(Q=Q, P=P, n=n, x0=x0, y0=y0, x_end=x_span[1], num_steps=n_steps)
    x_ark4, y_ark4 = adaptive_rk4(Q=Q, P=P, n=n, x0=x0, y0=y0, x_end=x_span[1])

    model = ODENet()
    model = train_ode_net(Q=Q, P=P, n=n, model=model, x_range=x_span, initial_condition=(x0, y0))
    x_nn = torch.linspace(x_span[0], x_span[1], n_steps + 1).view(-1, 1)
    y_nn = model(x_nn).detach().numpy()

    solution = solve_ivp(
        bernoulli_rhs,
        x_span,
        [y0],
        method='RK45',
        rtol=1e-8,
        atol=1e-8,
        dense_output=True
    )
    x_scipy = np.linspace(x0, x_span[1], n_steps + 1)
    y_scipy = solution.sol(x_scipy)[0]
    y_exact = exact_solution(x_vals)

    plt.plot(x_vals, y_euler, label="Euler's Method", linestyle='dotted')
    plt.plot(x_rk4, y_rk4, label="RK4 Method")
    plt.plot(x_ark4, y_ark4, label="Adaptive RK4 Method")
    plt.plot(x_nn, y_nn, label="Neural Network")
    plt.plot(x_vals, y_exact, label="Exact Solution", linestyle='dashed')
    plt.plot(x_vals, y_scipy, label="Scipy Solution", linestyle='dashdot')
    plt.legend()
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Comparison of ODE Solutions")
    plt.show()

    error_euler = np.mean(np.abs(exact_solution(x_euler) - y_euler))
    error_rk4 = np.mean(np.abs(exact_solution(x_rk4) - y_rk4))
    error_adaptive = np.mean(np.abs(exact_solution(x_ark4) - y_ark4))
    error_nn = np.mean(np.array(abs(exact_solution(x_nn) - y_nn)))
    error_scipy = np.mean(np.abs(exact_solution(x_scipy) - y_scipy))

    print(f"Mean absolute error (Euler): {error_euler:.6f}")
    print(f"Mean absolute error (RK4): {error_rk4:.6f}")
    print(f"Mean absolute error (Adaptive RK4): {error_adaptive:.6f}")
    print(f"Mean absolute error (Neural Network): {error_nn:.6f}")
    print(f"Mean absolute error (Scipy): {error_scipy:.6f}")

    methods = ["Euler", "RK4", "Adaptive RK4", "Neural Network", "Scipy"]
    errors = [error_euler, error_rk4, error_adaptive, error_nn, error_scipy]

    plt.figure(figsize=(8, 5))
    plt.bar(methods, errors, color=["blue", "green", "red", "purple", "orange"])

    plt.ylabel("Mean Absolute Error")
    plt.title("Comparison of Numerical Methods for Solving ODEs")
    plt.yscale("log")
    plt.show()
