import numpy as np

def step_function(h):
    return np.sign(h)

def d_identity(h):
    return 1

def db(t,C,output_set):
    if(t<1):
        return C*(-1)*np.sum(output_set)
    else:
        return 0

def dw(t,C,w,set,output_set):
    if (t<1):
        sum=0
        for (idx,example) in enumerate(set):
            sum += example*output_set[idx]
        return w + (C * (-1) * sum)
    else:
        return w