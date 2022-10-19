import numpy as np

def step_function(h):
    return np.sign(h)

def d_identity(h):
    return 1

def I(xi,yi,w,b):
    t = yi*(np.dot(w,xi) + b)
    if(t<1):
        return 1-t
    else:
        return 0

def L_error(w,b,training_set,output_set,C):
    result = (1/2) * np.linalg.norm(w)

    for (idx,entry) in enumerate(training_set):
        result+=C*I(entry,output_set[idx],w,b)
    
    return result

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