import numpy as np

def step_function(h):
    return np.sign(h)

def d_identity(h):
    return 1