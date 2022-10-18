import numpy as np

def step_function(h):
    return np.sign(h)

def d_identity(h):
    return 1

def db(t,C,output):
    if(t<1):
        return -C * output
    else:
        return 0

def dw(t,C,w,example,output):
    if (t<1):
        return w - (C * example * output)
    else:
        return w