"""Problem-wide constants for the 1D Burgers ROM workbench."""

import numpy as np

from .core import make_1d_grid

SEED = 1234557
SNAP_FOLDER = "param_snaps"
TIME_SCHEME = "backward_euler"

DT = 0.05
NUM_STEPS = 500
NUM_CELLS = 5000
XL, XU = 0.0, 100.0

GRID_X = make_1d_grid(XL, XU, NUM_CELLS)
CELL_CENTERS = 0.5 * (GRID_X[1:] + GRID_X[:-1])
U0 = np.ones(NUM_CELLS, dtype=np.float64)

MU1_RANGE = 4.25, 5.50
MU2_RANGE = 0.015, 0.030
SAMPLES_PER_MU = 3
