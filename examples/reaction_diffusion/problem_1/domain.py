from skfem import MeshTri, Basis, ElementTriP1
import numpy as np

"""
def domain_():
    print("hersssssse")
    mesh = MeshTri.load("problem_1.msh")


    # ------------------------------------------------------------
    # Reattach subdomain and boundary names explicitly.
    # This avoids relying on meshio/scikit-fem parsing the Gmsh tags.
    # ------------------------------------------------------------
    obstacle_boxes = [
        (1.0, 2.0, 5.0, 6.0),
        (5.0, 6.0, 5.0, 6.0),
        (2.0, 3.0, 4.0, 5.0),
        (4.0, 5.0, 4.0, 5.0),
        (1.0, 2.0, 3.0, 4.0),
        (5.0, 6.0, 3.0, 4.0),
        (2.0, 3.0, 2.0, 3.0),
        (4.0, 5.0, 2.0, 3.0),
        (1.0, 2.0, 1.0, 2.0),
        (3.0, 4.0, 1.0, 2.0),
        (5.0, 6.0, 1.0, 2.0),
    ]

    def in_any_obstacle(x):
        xx, yy = x[0], x[1]
        mask = np.zeros(xx.shape, dtype=bool)
        for xmin, xmax, ymin, ymax in obstacle_boxes:
            mask |= (
                (xx > xmin) & (xx < xmax) &
                (yy > ymin) & (yy < ymax)
            )
        return mask

    def in_source(x):
        xx, yy = x[0], x[1]
        return (
            (xx > 3.0) & (xx < 4.0) &
            (yy > 3.0) & (yy < 4.0)
        )

    def on_outer_boundary(x):
        xx, yy = x[0], x[1]
        tol = 1e-12
        return (
            np.isclose(xx, 0.0, atol=tol) |
            np.isclose(xx, 7.0, atol=tol) |
            np.isclose(yy, 0.0, atol=tol) |
            np.isclose(yy, 7.0, atol=tol)
        )

    mesh = (
        mesh
        .with_subdomains({
            "mat1": in_any_obstacle,                    # obstacles
            "mat2": in_source,                          # source
            "mat3": lambda x: ~(in_any_obstacle(x) | in_source(x)),  # background
        })
        .with_boundaries({
            "boundary": on_outer_boundary,
        })
    )

    element = ElementTriP1()
    basis = Basis(mesh, element)

    basis_mat1 = Basis(mesh, element, elements="mat1")
    basis_mat2 = Basis(mesh, element, elements="mat2")
    basis_mat3 = Basis(mesh, element, elements="mat3")

    dirichlet_boundary_dofs = basis.get_dofs("boundary")
    free_dofs = basis.complement_dofs(dirichlet_boundary_dofs)

    return {
        "mesh": mesh,
        "basis": basis,
        "basis_mat1": basis_mat1,
        "basis_mat2": basis_mat2,
        "basis_mat3": basis_mat3,
        "free_dofs": free_dofs,
        "dirichlet_boundary_dofs": dirichlet_boundary_dofs,
        "dirichlet_boundary_value": 0.0,
    }
"""

from skfem import MeshTri, Basis, ElementTriP1

def domain_():
    mesh = MeshTri.load("Problem_1.msh")
    element = ElementTriP1()
    basis = Basis(mesh, element)

    basis_mat1 = Basis(mesh, element, elements="mat1")
    basis_mat2 = Basis(mesh, element, elements="mat2")
    basis_mat3 = Basis(mesh, element, elements="mat3")

    dirichlet_boundary_dofs = basis.get_dofs("boundary")
    free_dofs = basis.complement_dofs(dirichlet_boundary_dofs)

    return {
        "mesh": mesh,
        "basis": basis,
        "basis_mat1": basis_mat1,
        "basis_mat2": basis_mat2,
        "basis_mat3": basis_mat3,
        "free_dofs": free_dofs,
        "dirichlet_boundary_dofs": dirichlet_boundary_dofs,
        "dirichlet_boundary_value": 0.0,
    }