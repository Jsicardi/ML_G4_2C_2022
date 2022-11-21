import numpy as np

class KohonenProperties:
    def __init__(self,input_ids,input_genres,input_set,eta,k,r,epochs):
        self.method = "kohonen"
        self.input_ids = input_ids
        self.input_genres = input_genres
        self.input_set = input_set
        self.eta = eta
        self.k = k
        self.r = r
        self.epochs = epochs
    
class KohonenObservables:
    def __init__(self,classifications,u_matrix,weights_matrix):
        self.classifications = classifications
        self.u_matrix = u_matrix
        self.weights_matrix = weights_matrix

class KohonenNeuron:
    def __init__(self,w,i,j):
        self.w = w
        self.i = i
        self.j = j
    
    def update_w(self,xp,eta):
        self.w += eta * (np.array(xp) - np.array(self.w))