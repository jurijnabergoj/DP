import numpy as np


def numericalDerivative(f, vec, h=1e-5):
    """
    Compute the Jacobian matrix of a vector-valued function using finite differences.

    Parameters:
        f : function
            A function that takes a vector and returns a vector.
        vec : array-like
            A point at which to compute the derivative.
        h : float, optional
            Step size for finite differences. Default is 1e-5.

    Returns:
        Jacobian : ndarray
            The Jacobian matrix of f at vec.
    """
    vec = np.array(vec, dtype=float)
    n = len(vec)  # input dim
    m = len(f(vec))  # output dim
    Jacobian = np.zeros((m, n))

    for i in range(n):
        vec_h = vec.copy()
        vec_h[i] += h
        f_plus = f(vec_h)

        vec_h[i] -= 2 * h
        f_minus = f(vec_h)

        Jacobian[:, i] = (np.array(f_plus) - np.array(f_minus)) / (2 * h)

    return Jacobian


# Test functions and their analytically computed Jacobians
def test_function1(vec):
    x, y = vec
    return [np.sin(x) + np.cos(y), np.cos(x) - np.sin(y)]


def analytical_jacobian1(vec):
    x, y = vec
    return np.array([
        [np.cos(x), -np.sin(y)],
        [-np.sin(x), -np.cos(y)]
    ])


def test_function2(vec):
    x, y = vec
    return [x ** 2 + y ** 2, x * y]


def analytical_jacobian2(vec):
    x, y = vec
    return np.array([
        [2 * x, 2 * y],
        [y, x]
    ])


def test_function3(vec):
    x, y = vec
    return [np.exp(x) * np.sin(y), np.log(1 + x ** 2 + y ** 2)]


def analytical_jacobian3(vec):
    x, y = vec
    return np.array([
        [np.exp(x) * np.sin(y), np.exp(x) * np.cos(y)],
        [2 * x / (1 + x ** 2 + y ** 2), 2 * y / (1 + x ** 2 + y ** 2)]
    ])


def test_function4(vec):
    x, y, z = vec
    return [x + y + z, x * y * z, np.sin(x) + np.cos(y) + np.exp(z)]


def analytical_jacobian4(vec):
    x, y, z = vec
    return np.array([
        [1, 1, 1],
        [y * z, x * z, x * y],
        [np.cos(x), -np.sin(y), np.exp(z)]
    ])


def test_function5(vec):
    x, y = vec
    return [np.tanh(x) + np.arctan(y), x / (1 + y ** 2)]


def analytical_jacobian5(vec):
    x, y = vec
    return np.array([
        [1 / np.cosh(x) ** 2, 1 / (1 + y ** 2)],
        [1 / (1 + y ** 2), -2 * x * y / (1 + y ** 2) ** 2]
    ])


def test_all_functions():
    test_cases = [
        ([np.pi / 4, np.pi / 3], test_function1, analytical_jacobian1),
        ([1.0, 2.0], test_function2, analytical_jacobian2),
        ([0.5, -0.5], test_function3, analytical_jacobian3),
        ([1.0, 2.0, 3.0], test_function4, analytical_jacobian4),
        ([0.1, 0.2], test_function5, analytical_jacobian5)
    ]

    # Calculate Jacobian and compare with analytical result for each test case
    for i, (vec, func, analytical_func) in enumerate(test_cases):
        jacobian_result = numericalDerivative(func, vec, h=0.0001)
        analytical_result = analytical_func(vec)
        diff = np.abs(jacobian_result - analytical_result)

        print(f"\n============\nTEST CASE {i}\n============\n")
        print(f"Numerical Jacobian at {vec}")
        print(jacobian_result)

        print("Analytical Jacobian:")
        print(analytical_result)

        print("Difference between numerical and analytical Jacobian:")
        print(diff)


if __name__ == "__main__":
    test_all_functions()
