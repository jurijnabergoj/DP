# Homework 2: Solving PDEs using Neural Networks #


## Requirements:
- Python >= 3.8
- numpy
- torch

In this assignment we implemented and compared
numerical and neural-network-based solutions of ODEs and PDEs. 
Included are 2 main python scripts which use methods in ./methods directory to solve differential equations:

`python .\solveODE.py` \
`python .\solvePDE.py`

## Solving ODEs ##
### Methods used ###
- Euler's method
- Runge-Kutta 4th order
- Adaptive Runge-Kutta 4th order (tried optimization)
- Neural network (a fully connected neural network with three hidden layers using the Tanh activation function)
- Scipy's ivp solver (just for comparison)

## Result visualization
The solutions of the ODEs are plotted on a graph as well as the exact (analytical) solution.
The mean absolute errors of each of these methods are displayed on a bar plot.

## Solving PDEs (Laplace equation) ##
- Finite difference method
- Neural network (a fully connected neural network with three hidden layers using the Tanh activation function)

## Result visualization
The solution obtained by each method is visualized on 4 plots: Contour, surface, stream and quiver plot.
