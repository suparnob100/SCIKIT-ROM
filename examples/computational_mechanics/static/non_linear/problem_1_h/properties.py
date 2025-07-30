import numpy as np  # array operations
from skfem.helpers import grad, identity, inv

def deformation_gradient(w):
    dudX = grad(w["displacement"])
    elem_indices = w.get('elem_indices', None)
    if elem_indices is None:
        F = dudX + identity(dudX)
        return F, inv(F)
    else:
        G = dudX[:,:, elem_indices, :] + identity(dudX[:,:, elem_indices, :])
        return G, inv(G)