# Domain dimensions and mesh resolution
import numpy as np

def material_properties():
    
    # Physical constants
    rho = 8e-3         # density
    cp = 0.5           # specific heat capacity
    k = 0.01           # thermal conductivity
    T_infty = 298.0    # ambient temperature
    h       = 2e-5      # convection coefficient (example)

    # Radiation
    Rboltz = 5.6704e-14
    emiss = 0.3

    return rho, cp, k, T_infty, h, Rboltz, emiss