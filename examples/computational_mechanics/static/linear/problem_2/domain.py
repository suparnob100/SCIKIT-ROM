import numpy as np
from scipy.spatial import Delaunay

from skfem import MeshTri, Basis, FacetBasis, ElementTriP1, ElementVector

"""
Domain: 2D plate with a circular hole (plane stress example)

Geometry
--------
Rectangle: [0, L] x [0, H]
Hole:      radius r centered at (L/2, H/2)

Boundary labels
---------------
- 'left'   : x = 0  (Dirichlet clamp u = 0)
- 'right'  : x = L  (Neumann traction in +x direction)
- 'top'    : y = H
- 'bottom' : y = 0
- 'hole'   : circular boundary (approx)

Mesh generation
---------------
Self-contained point-cloud + Delaunay triangulation, with elements inside the hole removed.

Robustness notes
----------------
1) We *prune unused vertices* after removing elements. If unused vertices remain, the stiffness
   matrix will have all-zero rows/cols for those DOFs, making the system singular (often showing
   up as NaNs in solutions).
2) We drop (near-)degenerate triangles to avoid division by zero in post-processing.
"""

def _make_point_cloud(L, H, r, *, nx=45, ny=25, n_circle=120, n_rings=3, seed=0):
    rng = np.random.default_rng(seed)

    # structured points on the rectangle (guarantees exact x=0 and x=L points)
    xs = np.linspace(0.0, L, nx)
    ys = np.linspace(0.0, H, ny)
    X, Y = np.meshgrid(xs, ys, indexing='xy')
    pts = np.column_stack([X.ravel(), Y.ravel()])

    cx, cy = 0.5 * L, 0.5 * H

    # remove points inside the hole (keep a thin buffer)
    d = np.sqrt((pts[:, 0] - cx) ** 2 + (pts[:, 1] - cy) ** 2)
    pts = pts[d >= 0.98 * r]

    # refined rings around the hole to improve boundary resolution
    thetas = np.linspace(0.0, 2.0 * np.pi, n_circle, endpoint=False)
    ring_r = [r] + [r * (1.15 + 0.15 * k) for k in range(n_rings)]
    ring_pts = []
    for rr in ring_r:
        ring_pts.append(np.column_stack([cx + rr * np.cos(thetas), cy + rr * np.sin(thetas)]))
    ring_pts = np.vstack(ring_pts)

    # additional random points (outside the hole)
    n_rand = max(200, nx * ny // 5)
    rand_pts = rng.random((n_rand, 2)) * np.array([L, H])
    d = np.sqrt((rand_pts[:, 0] - cx) ** 2 + (rand_pts[:, 1] - cy) ** 2)
    rand_pts = rand_pts[d >= 1.05 * r]

    pts = np.vstack([pts, ring_pts, rand_pts])

    # remove duplicates (round for stability)
    pts = np.unique(np.round(pts, decimals=12), axis=0)

    return pts

def _prune_unused_vertices(pts, t):
    """Prune unused vertices and remap connectivity."""
    used = np.unique(t.ravel())
    new_id = -np.ones(pts.shape[0], dtype=int)
    new_id[used] = np.arange(len(used))
    pts_new = pts[used]
    t_new = new_id[t]
    return pts_new, t_new

def _drop_degenerate_triangles(pts, t, *, eps=1e-14):
    """Drop triangles with near-zero signed area."""
    p = pts
    a = p[t[:, 0]]
    b = p[t[:, 1]]
    c = p[t[:, 2]]
    twoA = (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (c[:, 0] - a[:, 0]) * (b[:, 1] - a[:, 1])
    keep = np.abs(twoA) > eps
    return t[keep]

def _triangulate_with_hole(pts, L, H, r):
    cx, cy = 0.5 * L, 0.5 * H

    tri = Delaunay(pts)
    t = tri.simplices.copy()  # (n_elems, 3)

    # remove triangles whose centroid lies inside the hole
    cent = pts[t].mean(axis=1)
    dist = np.sqrt((cent[:, 0] - cx) ** 2 + (cent[:, 1] - cy) ** 2)
    keep = dist >= 1.001 * r
    t = t[keep]

    # drop degenerate triangles (important for stress post-processing)
    t = _drop_degenerate_triangles(pts, t, eps=1e-14 * (L * H))

    # prune unused vertices (IMPORTANT: prevents singular K and NaNs)
    pts, t = _prune_unused_vertices(pts, t)

    # build mesh: p:(2,npts), t:(3,nelems)
    mesh = MeshTri(pts.T, t.T)

    # boundary tags
    tol = min(L, H) * 1e-10
    rtol = 0.03 * r  # tolerance for approximate hole boundary

    def _left(x):   return np.isclose(x[0], 0.0, atol=tol)
    def _right(x):  return np.isclose(x[0], L,   atol=tol)
    def _bottom(x): return np.isclose(x[1], 0.0, atol=tol)
    def _top(x):    return np.isclose(x[1], H,   atol=tol)

    def _hole(x):
        rr = np.sqrt((x[0] - cx) ** 2 + (x[1] - cy) ** 2)
        return np.abs(rr - r) <= rtol

    mesh = mesh.with_boundaries({
        'left': _left,
        'right': _right,
        'bottom': _bottom,
        'top': _top,
        'hole': _hole,
    })

    return mesh

def domain_(L=4.0, H=2.0, r=0.35, factor=1, dirichlet_boundary_value=0.0):
    """
    Return mesh + FEM bases for the plate-with-hole tension problem.

    Returns
    -------
    dict (keys consistent with other examples)
      - mesh, basis, element
      - fbasis_dirichlet, fbasis_neumann
      - dirichlet_dofs, neumann_dofs
      - dirichlet_boundary_value
      - basis_regions  (single region)
      - L, H, r        (geometry scalars)
    """
    pts = _make_point_cloud(L, H, r, nx=45 * factor, ny=25 * factor, n_circle=120, n_rings=3, seed=0)
    mesh = _triangulate_with_hole(pts, L, H, r).refined(2)  # optional uniform refinement for better accuracy

    # vector-valued P1 (linear triangles)
    element = ElementVector(ElementTriP1())
    basis = Basis(mesh, element)

    # facet indices from mesh.boundaries
    facets_left = mesh.boundaries['left']
    facets_right = mesh.boundaries['right']

    fbasis_dirichlet = FacetBasis(mesh, element, facets=facets_left)
    fbasis_neumann   = FacetBasis(mesh, element, facets=facets_right)

    # DOF sets
    dirichlet_dofs = basis.get_dofs('left')
    # sanity checks (avoid silent singular systems)
    if facets_left.size == 0:
        raise RuntimeError("No 'left' boundary facets found; check mesh generation/tolerances.")
    if facets_right.size == 0:
        raise RuntimeError("No 'right' boundary facets found; check mesh generation/tolerances.")
    if dirichlet_dofs.all().size == 0:
        raise RuntimeError("No Dirichlet DOFs found on 'left'; stiffness will be singular.")
    neumann_dofs   = basis.get_dofs('right')

    # single region (uniform material)
    basis_regions = {'region_1': basis}

    return {
        'mesh': mesh,
        'basis': basis,
        'element': element,
        'fbasis_dirichlet': fbasis_dirichlet,
        'fbasis_neumann': fbasis_neumann,
        'dirichlet_dofs': dirichlet_dofs,
        'neumann_dofs': neumann_dofs,
        'dirichlet_boundary_value': dirichlet_boundary_value,
        'basis_regions': basis_regions,
        'L': L, 'H': H, 'r': r,
    }
