"""\
Linear form (source term) for 2-D transient heat conduction.

We use a smooth Gaussian volumetric heat source centered at (0.5, 0.5).
The amplitude is parameterized in properties.py.
"""

import numpy as np
from skfem import LinearForm


@LinearForm
def l_source(v, w):
    x = w.x[0]
    y = w.x[1]

    # Gaussian source
    x0, y0 = 0.5, 0.5
    sigma = 0.12
    g = np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (2.0 * sigma ** 2))

    return g * v
