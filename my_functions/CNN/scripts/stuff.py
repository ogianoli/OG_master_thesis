import numpy as np
from master_thesis.my_functions.helper_functions import *
import torch
import math
print(torch.backends.mps.is_available())
print(torch.backends.mps.is_built())


    

if __name__ == "__main__":
    # === STAR Geometry Derived Inputs ===
    diameter = 0.0793          # Outer diameter [m]
    pointLength = 0.0225      # Length of one star point [m]
    pointWidth = 0.0006      # Width of one star point [m]
    numPoints = 1           # Number of points (N)

    # === STAR Geometry Parameters ===
    # N = int(numPoints)
    # Ro = diameter / 2
    # w = pointLength
    # Ri = Ro - w
    # f = 0.0  # Fillet radius (set to 0 or define rule if needed)
    # e=1
    # e = (2 / np.pi) * np.arctan(pointWidth / (2 * w))  # Derive angle coefficient from pointWidth and web thickness

    # === BATES Geometry Parameters ===
    bates_diameter = 0.06  # Outer diameter [m]
    bates_core_diameter = 0.02  # Core diameter [m]

    # === Endburner Geometry Parameters ===
    endburner_diameter = 0.05  # Diameter [m]

    # helper_functions.plot_star_geometry(N, Ri, Ro, w, f, e)
    # helper_functions.plot_star_petal(N, Ri, Ro, w, f, e)
    # helper_functions.plot_bates_geometry(bates_diameter, bates_core_diameter)
    # helper_functions.plot_endburner_geometry(endburner_diameter)
    generate_star_geometry_from_scratch_2(diameter, pointLength, pointWidth, numPoints)
    # === Example: Use new star plotting function from basic parameters ===
    # helper_functions.plot_star_direct(
    #     diameter=diameter,
    #     pointLength=pointLength,
    #     pointWidth=pointWidth,
    #     numPoints=numPoints
    # )