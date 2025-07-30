import numpy as np                          # core numerical arrays
from skfem import BilinearForm             # decorator for bilinear forms
from properties import deformation_gradient              # conductivity and source functions
from skfem.helpers import grad, identity, ddot, det, transpose, inv, trace, mul


@BilinearForm
def J_form(u, v, w):
    mu = w.get('mu')
    lmbda = w.get('lmbda')

    F, iF = deformation_gradient(w)
    DF = grad(u)
    dF = grad(v)
    dFiF = mul(dF, iF)
    DFiF = mul(DF, iF)
    tr_DFiF_dFiF = ddot(transpose(dFiF), DFiF)
    lnJ = np.log(det(F))
    return (mu * ddot(DF, dF) - (lmbda * lnJ - mu) * tr_DFiF_dFiF
            + lmbda * trace(dFiF) * trace(DFiF))