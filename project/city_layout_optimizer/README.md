# City Layout Optimization in Blender

This project implements a differentiable system for optimizing procedural city layouts using Blender and Python. It allows users to optimize building positions based on urban design objectives such as road avoidance, amenity proximity, building separation, and sunlight exposure. The optimization is implemented as a Blender add-on using SciPy's SLSQP optimizer.


## Getting Started

### 1. Install Dependencies

You must install the required Python packages inside Blender’s bundled Python environment:

```bash
# Locate Blender's Python binary (example)
cd /path/to/blender/<version>/python/bin

# Install pip if needed
./python3.10 -m ensurepip

# Install required packages
./python3.10 -m pip install torch scipy
```

### 2. Install the Add-on
You can install the add-on by following these instructions:

- Zip the file containing all python scripts or use the standalone city_optimizer.py or differentiable_optimizer.py
- Open Blender
- Go to Edit > Preferences > Add-ons 
- Click Install and select the zip or standalone file 
- Enable the add-on by checking the box

### 3. Load a Scene

- Open Blender
- Navigate to Scripting > Text > Open 
- Load the python scene file from the /scenes folder 
- Open the City Optimizer panel in the sidebar
- Click Run Optimization to refine the layout


### Project Structure
```bash
city-layout-optimizer/
├── scripts/
│   ├── city_optimizer.py               # Non-differentiable optimizer
│   ├── differentiable_optimizer.py     # Differentiable optimizer
├── optimizer_module/                   # Work in progress refactor of the python files under /scripts
│   ├── __init__.py                       # Initializer
│   ├── geometry_utils.py                 # Utility functions
│   ├── optimizer.py                      # Optimizer
│   ├── scene_utils.py                    # Utility functions
│── scenes/
│   └──scene_ring.py                    # Prebuilt test scene
│   └──scene_sun.py                     # Prebuilt test scene
└── README.md
```
