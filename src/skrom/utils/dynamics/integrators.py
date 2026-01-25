import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import splu


def newmark_with_damping(M, C, K, force_free, times, U0=None, V0=None, beta=0.25, gamma=0.5):
    M_mat = M.copy() if hasattr(M, "copy") else sp.csr_matrix(M)
    C_mat = C.copy() if hasattr(C, "copy") else sp.csr_matrix(C)
    K_mat = K.copy() if hasattr(K, "copy") else sp.csr_matrix(K)

    N = len(times)
    dt = times[1] - times[0]  # assumes uniform spacing
    n = M_mat.shape[0]

    a0 = 1.0 / (beta * dt**2)
    a1 = gamma / (beta * dt)
    a2 = 1.0 / (beta * dt)
    a3 = 1.0 / (2 * beta) - 1.0
    a4 = gamma / beta - 1.0
    a5 = dt * (gamma / (2 * beta) - 1.0)
    a6 = dt * (1 - gamma)
    a7 = gamma * dt

    def _flat(v):
        return v.A1 if hasattr(v, "A1") else np.asarray(v).ravel()

    # Factor M (CSC)
    M_csc = M_mat.tocsc() if sp.issparse(M_mat) else sp.csc_matrix(M_mat)
    M_fac = splu(M_csc)

    # Factor K_eff (CSC)
    K_eff = (K_mat + a0 * M_mat + a1 * C_mat)
    K_csc = K_eff.tocsc() if sp.issparse(K_eff) else sp.csc_matrix(K_eff)
    K_fac = splu(K_csc)

    U = np.zeros((n, N))
    V = np.zeros_like(U)
    A = np.zeros_like(U)

    if U0 is not None:
        U[:, 0] = _flat(U0)
    if V0 is not None:
        V[:, 0] = _flat(V0)

    F0 = _flat(force_free(0, times))
    r0 = F0 - _flat(C_mat.dot(V[:, 0])) - _flat(K_mat.dot(U[:, 0]))
    A[:, 0] = M_fac.solve(_flat(r0))

    for i in range(N - 1):
        F_np1 = _flat(force_free(i + 1, times))
        R_hat = (
            F_np1
            + _flat(M_mat.dot(a0 * U[:, i] + a2 * V[:, i] + a3 * A[:, i]))
            + _flat(C_mat.dot(a1 * U[:, i] + a4 * V[:, i] + a5 * A[:, i]))
        )
        U[:, i + 1] = K_fac.solve(_flat(R_hat))

        A[:, i + 1] = a0 * (U[:, i + 1] - U[:, i]) - a2 * V[:, i] - a3 * A[:, i]
        V[:, i + 1] = V[:, i] + a6 * A[:, i] + a7 * A[:, i + 1]

    return U, V, A


def wbz_alpha(M, C, K, force_free, times,
              U0=None, V0=None,
              gamma=0.5, beta=0.25, alpha_m=0.0):
    """
    Wood-Bossak-Zienkiewicz (WBZ-α) method.
    Extension of Newmark method with algorithmic damping parameter alpha_m.
    alpha_m: numerical damping parameter (0 ≤ alpha_m ≤ 1)
    """
    M_mat = M.copy() if hasattr(M, 'copy') else sp.csr_matrix(M)
    C_mat = C.copy() if hasattr(C, 'copy') else sp.csr_matrix(C)
    K_mat = K.copy() if hasattr(K, 'copy') else sp.csr_matrix(K)

    N, dt, n = len(times), times[1] - times[0], M_mat.shape[0]

    # WBZ-α coefficients
    a0 = 1.0 / (beta * dt**2)
    a1 = gamma / (beta * dt)
    a2 = 1.0 / (beta * dt)
    a3 = 1.0 / (2.0 * beta) - 1.0
    a4 = gamma / beta - 1.0
    a5 = dt * (gamma / (2.0 * beta) - 1.0)
    a6 = dt * (1.0 - gamma)
    a7 = dt * gamma

    # Factor mass matrix
    M_csr = M_mat if sp.issparse(M_mat) else sp.csr_matrix(M_mat)
    M_fac = splu(M_csr.copy())

    # Effective stiffness with WBZ-α modification
    K_eff = K_mat + a0 * M_mat + a1 * C_mat
    K_fac = splu(K_eff if sp.issparse(K_eff) else sp.csr_matrix(K_eff))

    def _flat(v):
        return v.A1 if hasattr(v, "A1") else np.asarray(v).ravel()

    U = np.zeros((n, N))
    V = np.zeros_like(U)
    A = np.zeros_like(U)

    if U0 is not None:
        U[:, 0] = np.array(U0).copy()
    if V0 is not None:
        V[:, 0] = np.array(V0).copy()

    # Initial acceleration
    F0 = force_free(0, times)
    r0 = F0 - C_mat.dot(V[:, 0]) - K_mat.dot(U[:, 0])
    A[:, 0] = M_fac.solve(_flat(r0))

    for i in range(N - 1):
        F_np1 = force_free(i + 1, times)

        # WBZ-α effective RHS with mass weighting
        R_hat = (
            F_np1
            + M_mat.dot(a0 * U[:, i] + a2 * V[:, i] + a3 * A[:, i])
            + C_mat.dot(a1 * U[:, i] + a4 * V[:, i] + a5 * A[:, i])
            - alpha_m * M_mat.dot(A[:, i])
        )

        U[:, i + 1] = K_fac.solve(_flat(R_hat))

        # Modified acceleration update with alpha_m damping
        A[:, i + 1] = (
            a0 * (U[:, i + 1] - U[:, i])
            - a2 * V[:, i]
            - a3 * A[:, i]
        ) / (1 + alpha_m)

        V[:, i + 1] = V[:, i] + a6 * A[:, i] + a7 * A[:, i + 1]

    return U, V, A


