import numpy as np

class HierarchicalProperties:
    def __init__(self,input_ids,input_genres,input_set,distance_method,k):
        self.method = "hierarchical"
        self.input_ids = input_ids
        self.input_genres = input_genres
        self.input_set = input_set
        self.distance_method = distance_method
        self.k = k

class HierarchicalObservables:
    def __init__(self,groups):
        self.method = "hierarchical"
        self.groups = groups

class HierarchicalGroup:
    def __init__(self,members,members_genres):
        self.members = members
        self.members_genres = members_genres

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