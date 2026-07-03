"""Default settings for the KdV soliton benchmark from Geelen et al."""

import numpy as np

ALPHA = 4.0
BETA = 1.0

XL = -np.pi
XU = np.pi
NUM_POINTS = 256

DT = 2.0e-4
FINAL_TIME = 1.0
TRAIN_FINAL_TIME = 0.2

ETDRK4_CONTOUR_POINTS = 32
DEALIAS = True


def make_periodic_grid(num_points=NUM_POINTS, xl=XL, xu=XU):
    """Return an endpoint-excluding periodic grid on [xl, xu)."""
    return np.linspace(float(xl), float(xu), int(num_points), endpoint=False, dtype=np.float64)


def initial_condition(x):
    """Initial condition s0(x) = 1 + 24 sech^2(sqrt(8) x)."""
    x = np.asarray(x, dtype=np.float64)
    return 1.0 + 24.0 / np.cosh(np.sqrt(8.0) * x) ** 2

