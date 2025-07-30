import numpy as np  # array operations
from skfem.helpers import grad, identity, inv

def E(mu):
    return 77e3 + mu

def rho(rho_param):
    return 0.28907 + rho_param