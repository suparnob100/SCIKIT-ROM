from skfem import LinearForm

"""
Neumann traction for plate-with-hole tension test.

Applies uniform traction in +x direction on the 'right' boundary:

  t = (t0, 0)

You can override t0 via assemble kwargs (t0=...); default is t0=1.0.
"""


@LinearForm
def traction(v, w):
    t0 = float(w.get('t0', 1.0))
    return t0 * v[0]
