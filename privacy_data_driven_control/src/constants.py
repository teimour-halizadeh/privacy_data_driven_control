import numpy as np



"""
This module contains all constants used for simulation.
"""

# system model in the paper
As = np.array([[1.178, 0.001, 0.511, -0.403],
                [-0.051, 0.661, -0.011, 0.061],
                [0.076, 0.335, 0.560, 0.382],
                [0, 0.335, 0.089, 0.849]])
Bs = np.array([[0.004, -0.087],
                [0.467, 0.001],
                [0.213, -0.235],
                [0.213, -0.016]])




# blue ellipsoid parameters
A_INV_12 = np.diag([1, 0.05]) 
ZETA = np.array([[1],[1]]) 
GAMMA_VAL= 0.03 # offset value
TRUE_SYSTEM = ZETA + np.array([[0.01],[0.04]])


# nominal system parameters
T = 20
INPUT_RANGE = 5
INITIAL_STATE_RANGE = 2.5
KEY_RANGE = 1
MAX_DISTURBANCE = 0  

# Parameters for injection bias
SIM_INIT_STATE_RANGE = 0.1
SIM_TIME = 30
THRESHOLD = 0.2
T_INJECTION = 10
BETA = 0.5

# Data set parameters for disturbance gamma relation
NUM_DATA_SETS = 10**3
DIST_RANGE = [0, 0.02, 0.04, 0.06, 0.08, 0.1, 0.12, 0.14, 0.16]
GAMMA_RANGE = np.arange(0,0.1,0.001)