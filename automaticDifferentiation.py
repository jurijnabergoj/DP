import math
from typing import Dict, Optional


class DiffVar:
    """
    A differentiable variable that stores its value and derivatives with respect to independent variables.
    """

    def __init__(self, value: float, derivatives: Optional[Dict[str, float]] = None, name: str = None):
        self.value = value
        self.derivatives = derivatives or {}
        self.name = name

        # If this is an independent variable, set its derivative with respect to itself to 1
        if name is not None and name not in self.derivatives:
            self.derivatives[name] = 1.0

    def __repr__(self):
        return f"DiffVar(value={self.value}, derivatives={self.derivatives}, name={self.name})"

    # Addition
    # d(f+g)/dx = df/dx + dg/dx
    def __add__(self, other):
        if not isinstance(other, DiffVar):
            other = DiffVar(float(other))

        all_variables = set(self.derivatives) | set(other.derivatives)
        result_value = self.value + other.value
        result_derivatives = {var: self.derivatives.get(var, 0.0) + other.derivatives.get(var, 0.0)
                              for var in all_variables}

        return DiffVar(result_value, result_derivatives)

    def __radd__(self, other):
        return self + other

    # Multiplication
    # d(f*g)/dx = df/dx * g + f * dg/dx
    def __mul__(self, other):
        if not isinstance(other, DiffVar):
            other = DiffVar(float(other))

        all_variables = set(self.derivatives) | set(other.derivatives)
        result_value = self.value * other.value
        result_derivatives = {}
        for var in all_variables:
            df_dx = self.derivatives.get(var, 0.0)
            dg_dx = other.derivatives.get(var, 0.0)
            result_derivatives[var] = df_dx * other.value + self.value * dg_dx

        return DiffVar(result_value, result_derivatives)

    def __rmul__(self, other):
        return self * other

    # Power
    # d(f^n)/dx = n * f^(n-1) * df/dx
    def __pow__(self, power):
        if not isinstance(power, (int, float)):
            raise TypeError("Power must be a scalar value for this implementation")
        result_value = self.value ** power

        result_derivatives = {}
        for var, df_dx in self.derivatives.items():
            result_derivatives[var] = power * (self.value ** (power - 1)) * df_dx

        return DiffVar(result_value, result_derivatives)


# Sine
# d(sin(f))/dx = cos(f) * df/dx
def sin(x: DiffVar):
    result_value = math.sin(x.value)

    result_derivatives = {}
    for var, df_dx in x.derivatives.items():
        result_derivatives[var] = math.cos(x.value) * df_dx

    return DiffVar(result_value, result_derivatives)


# Cosine
# d(cos(f))/dx = -sin(f) * df/dx
def cos(x: DiffVar):
    result_value = math.cos(x.value)

    result_derivatives = {}
    for var, df_dx in x.derivatives.items():
        result_derivatives[var] = -math.sin(x.value) * df_dx

    return DiffVar(result_value, result_derivatives)


# Function to create an independent variable
def var(name: str, value: float) -> DiffVar:
    """Create an independent variable with the given name and value."""
    return DiffVar(value, {name: 1.0}, name)


# Function to print results
def print_results(f_name: str, f: DiffVar, vars_dict: Dict[str, DiffVar]):
    print(f"=== Testing {f_name} ===")
    var_values = ", ".join([f"{var.name}={var.value}" for var in vars_dict.values()])
    print(f"Function values at {var_values}:")
    print(f"  f({var_values}) = {f.value}")

    print("Partial derivatives:")
    for var_name in vars_dict:
        if var_name in f.derivatives:
            print(f"  df/d{var_name} = {f.derivatives[var_name]}")


# Test functions
def test_all_functions():
    test_cases = [
        {
            "name": "f(x, y) = 3x + 4y + 5",
            "function": lambda x, y: 3 * x + 4 * y + 5,
            "variables": {"x": 2.0, "y": 3.0},
            "expected": {
                "value": 3 * 2.0 + 4 * 3.0 + 5,
                "derivatives": {"x": 3.0, "y": 4.0}
            }
        },
        {
            "name": "f(x, y) = 3xy + 5",
            "function": lambda x, y: 3 * x * y + 5,
            "variables": {"x": 2.0, "y": 3.0},
            "expected": {
                "value": 3 * 2.0 * 3.0 + 5,
                "derivatives": {"x": 3 * 3.0, "y": 3 * 2.0}
            }
        },
        {
            "name": "f(x, y, z) = 5x + 3y + 4xyz",
            "function": lambda x, y, z: 5 * x + 3 * y + 4 * x * y * z,
            "variables": {"x": 1.0, "y": 2.0, "z": 3.0},
            "expected": {
                "value": 5 * 1.0 + 3 * 2.0 + 4 * 1.0 * 2.0 * 3.0,
                "derivatives": {
                    "x": 5 + 4 * 2.0 * 3.0,
                    "y": 3 + 4 * 1.0 * 3.0,
                    "z": 4 * 1.0 * 2.0
                }
            }
        },
        {
            "name": "f(x, y) = 2sin(x) + 3cos(y)",
            "function": lambda x, y: 2 * sin(x) + 3 * cos(y),
            "variables": {"x": 0.5, "y": 1.0},
            "expected": {
                "value": 2 * math.sin(0.5) + 3 * math.cos(1.0),
                "derivatives": {
                    "x": 2 * math.cos(0.5),
                    "y": -3 * math.sin(1.0)
                }
            }
        },
        {
            "name": "f(x, y) = x^2 * y^3",
            "function": lambda x, y: pow(x, 2) * pow(y, 3),
            "variables": {"x": 2.0, "y": 3.0},
            "expected": {
                "value": (2.0 ** 2) * (3.0 ** 3),
                "derivatives": {
                    "x": 2 * 2.0 * (3.0 ** 3),
                    "y": (2.0 ** 2) * 3 * (3.0 ** 2)
                }
            }
        }
    ]

    for i, test_case in enumerate(test_cases):
        # Create variables
        vars_dict = {name: var(name, value) for name, value in test_case["variables"].items()}

        # Evaluate function with automatic differentiation
        f = test_case["function"](**vars_dict)

        # Print results
        print(f"\n============\nTEST CASE {i}\n============\n")
        print_results(test_case["name"], f, vars_dict)

        # Verify that results are correct
        expected = test_case["expected"]
        threshold = 1e-10  # Value is correct if error is under threshold
        print("Verification:")
        print(f"  Expected value: {expected['value']}")
        print(f"  Computed value: {f.value}")
        print(f"  Value correct: {abs(f.value - expected['value']) < threshold}")

        for var_name, expected_deriv in expected["derivatives"].items():
            actual_deriv = f.derivatives.get(var_name, 0.0)
            print(f"  Expected df/d{var_name}: {expected_deriv}")
            print(f"  Computed df/d{var_name}: {actual_deriv}")
            print(f"  Derivative correct: {abs(actual_deriv - expected_deriv) < threshold}")


if __name__ == "__main__":
    test_all_functions()
