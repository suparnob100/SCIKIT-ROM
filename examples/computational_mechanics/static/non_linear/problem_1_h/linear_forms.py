import numpy as np                        # numerical arrays
from properties import deformation_gradient            # conductivity/source functions
from skfem.helpers import grad, dot       # gradient and dot-product
from skfem import LinearForm              # decorator for residual forms
from skfem.helpers import grad, identity, ddot, det, transpose, inv, trace, mul

@LinearForm
def R(v, w):
    mu = w.get('mu')
    lmbda = w.get('lmbda')
    F, iF = deformation_gradient(w)
    dF = grad(v)
    lnJ = np.log(det(F))
    return mu * ddot(F, dF) + (lmbda * lnJ - mu) * ddot(transpose(iF), dF)