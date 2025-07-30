"""
Implements bounded non-negative least squares (NNLS) for Empirical Cubature Subset Weighting (ECSW).

This module provides:
  - `NNLS_termination`: enumeration of L2 and L∞ convergence criteria for NNLS.
  - `_verify`: internal helper to assert solver invariants.
  - `NNLSSolver`: a sequential active-set NNLS solver with per-entry bounds, selectable norms,
    stall-detection, and verbosity controls.

The `ecsw` folder contains utilities for Empirical Cubature Subset Weighting, including:
  - Algorithms to compute cubature weights using bounded NNLS.
  - Selection and pruning of integration points via active-set methods.
  - Support functions for convergence criteria and solver configuration.
"""

import numpy as np
import scipy.linalg as sla  # For QR and triangular solve
import sys  # For flushing output
import enum

# --- Enums and Constants ---
class NNLS_termination(enum.Enum):
    """
    Termination criteria for the NNLS solver.

    Enumeration of the two supported norms used to decide convergence.

    Attributes
    ----------
    L2 : int
        Use the L₂-norm of the residual (‖r‖₂) compared against the half-gap
        norm threshold (‖(rhs_ub – rhs_lb)/2‖₂).
    LINF : int
        Use the L∞-norm criterion, i.e. the maximum per-entry violation must be
        no greater than the absolute tolerance (`const_tol`).
    """
    L2 = 0
    LINF = 1

def _verify(condition, message="Verification failed"):
    """
    Assert that `condition` is True.

    Parameters
    ----------
    condition : bool
        The condition to test.
    message : str, optional
        Error message to raise if `condition` is False (default:
        "Verification failed").

    Raises
    ------
    AssertionError
        If `condition` is False.
    """
    assert condition, message


# --- NNLSSolver Class (Sequential Version) ---
class NNLSSolver:
    """
    Sequential bounded NNLS (non-negative least squares) solver.

    Implements an active-set method for finding x ≥ 0 that approximately satisfies
    A x ≈ b, with per-entry bounds on b and two convergence tests (L₂‐ and L∞‐norm).

    Parameters
    ----------
    const_tol : float, optional
        Tolerance for constraint violation in the L∞‐criterion (default: 1e-6).
    min_nnz : int, optional
        Minimum number of nonzeros required in the solution before stopping
        (default: 1).
    max_nnz : int, optional
        Maximum allowed number of nonzeros in the solution.  A value of 0
        means “no limit” and will be set to the number of columns of A
        on the first `solve` call (default: 0).
    verbosity : int, optional
        Print level (0: silent, 1: summary only, ≥2: detailed per‐iteration logging)
        (default: 1).
    res_change_termination_tol : float, optional
        If the relative change in the mean residual over 50 iterations falls
        below this threshold, the solver will deem itself stalled (default: 1e-10).
    zero_tol : float, optional
        Threshold below which computed subproblem entries are considered zero
        (default: 1e-15).
    n_outer : int, optional
        Maximum number of outer (active‐set) iterations (default: 1000).
    n_inner : int, optional
        Maximum number of inner (subproblem) iterations per active set (default: 400).
    criterion : {NNLS_termination.L2, NNLS_termination.LINF}, optional
        Which norm to use for stopping test: L2 uses ‖r‖₂ ≤ ‖gap‖₂, L∞ uses
        max_violation ≤ const_tol  (default: L∞).

    Attributes
    ----------
    const_tol_ : float
        As given by `const_tol`.
    min_nnz_ : int
        As given by `min_nnz`.
    max_nnz_ : int
        As given by `max_nnz` or set at solve‐time.
    verbosity_ : int
        As given by `verbosity`.
    res_change_termination_tol_ : float
        As given by `res_change_termination_tol`.
    zero_tol_ : float
        As given by `zero_tol`.
    n_outer_ : int
        As given by `n_outer`.
    n_inner_ : int
        As given by `n_inner`.
    d_criterion : NNLS_termination
        As given by `criterion`.
    
    Examples
    --------
    >>> from nnls_solver import NNLSSolver, NNLS_termination
    >>> import numpy as np
    >>> A = np.random.rand(20, 10)
    >>> const_tol_ = 1e-3
    >>> lb = b - const_tol_
    >>> ub = b + const_tol_
    >>> solver = NNLSSolver(const_tol=const_tol_, verbosity=2)
    >>> x, flag = solver.solve(A, lb, ub)
    >>> print("Exit flag:", flag)
    """

    def __init__(self, const_tol=1e-6, min_nnz=1, max_nnz=0, verbosity=1,
                 res_change_termination_tol=1e-10, zero_tol=1e-15,
                 n_outer=1000, n_inner=400,
                 criterion=NNLS_termination.LINF):

        self.const_tol_ = float(const_tol)
        self.min_nnz_ = int(min_nnz)
        self.max_nnz_ = int(max_nnz)  # 0 means set later based on matrix size
        self.verbosity_ = int(verbosity)
        self.res_change_termination_tol_ = float(res_change_termination_tol)
        self.zero_tol_ = float(zero_tol)
        self.n_outer_ = int(n_outer)
        self.n_inner_ = int(n_inner)
        self.d_criterion = criterion

        _verify(self.d_criterion in [NNLS_termination.L2, NNLS_termination.LINF])

        if self.verbosity_ > 0:
            print("NNLSSolver init (Sequential Version)")

    def set_verbosity(self, verbosity_in):
        """
        Set the verbosity level.

        Parameters
        ----------
        verbosity_in : int
            New verbosity level (0: silent, larger for more output).
        """
        self.verbosity_ = int(verbosity_in)

    def solve(self, mat, rhs_lb, rhs_ub):
        """
        Solve A x ≈ b with 0 ≤ x and b∈[rhs_lb, rhs_ub] by active‐set NNLS.

        Parameters
        ----------
        mat : array_like, shape (m, n)
            Left‐hand‐side matrix A.
        rhs_lb : array_like, shape (m,)
            Per‐entry lower bounds on b.
        rhs_ub : array_like, shape (m,)
            Per‐entry upper bounds on b.

        Returns
        -------
        final_soln : ndarray, shape (n,)
            Computed nonnegative solution.
        exit_flag : int
            Status code:
              - 0: converged successfully
              - 1: maximum outer iterations reached
              - 2: stalled (no significant residual change)
              - 3: other failure (e.g., subproblem failure or M≤N).
        """
        m, n_tot = mat.shape
        soln = np.zeros(n_tot)
        _verify(rhs_lb.size == m and rhs_ub.size == m)

        if self.max_nnz_ == 0:
            self.max_nnz_ = n_tot

        rhs_avg = (rhs_ub + rhs_lb) * 0.5
        rhs_halfgap = (rhs_ub - rhs_lb) * 0.5
        l2norm_threshold = 0.0
        if self.d_criterion == NNLS_termination.L2:
            l2norm_threshold = np.linalg.norm(rhs_halfgap)
            if self.verbosity_ > 1:
                print(f"L2 norm threshold: {l2norm_threshold:.5e}")

        P_set = set()
        res = np.copy(rhs_avg)
        l2_res_hist = np.zeros(self.n_outer_)
        stalled_indices = set()
        exit_flag = 1
        n_total_inner_iter = 0

        for oiter in range(self.n_outer_):
            n_glob = len(P_set)
            P_idx = sorted(list(P_set))

            current_soln = np.zeros(n_tot)
            if n_glob > 0:
                current_soln[P_idx] = soln[P_idx]
            res = rhs_avg - mat @ current_soln

            abs_res = np.abs(res)
            violations = abs_res - rhs_halfgap
            rmax = np.max(violations) if violations.size > 0 else -np.inf
            l2_res_hist[oiter] = np.linalg.norm(res)

            if self.d_criterion == NNLS_termination.LINF:
                tolerance_met = (rmax <= self.const_tol_)
            else:
                tolerance_met = (l2_res_hist[oiter] <= l2norm_threshold)

            if self.verbosity_ > 1:
                print(f"{oiter} {n_total_inner_iter} {m} {n_tot} {n_glob} "
                      f"{rmax:.15e} {l2_res_hist[oiter]:.15e}")
                sys.stdout.flush()

            if tolerance_met and n_glob >= self.min_nnz_:
                if self.verbosity_ > 1:
                    print("Target tolerance met")
                exit_flag = 0
                break
            if n_glob >= self.max_nnz_:
                if self.verbosity_ > 1:
                    print("Target nnz met")
                exit_flag = 0
                break
            if n_glob >= m:
                if self.verbosity_ > 1:
                    print("System is square or underdetermined... exiting")
                exit_flag = 3
                break

            if oiter > 101:
                mean0 = np.mean(l2_res_hist[oiter - 50:oiter])
                mean1 = np.mean(l2_res_hist[oiter - 101:oiter - 51])
                if mean0 > 1e-15:
                    mean_res_change = (mean1 / mean0) - 1.0
                    if abs(mean_res_change) < self.res_change_termination_tol_:
                        if self.verbosity_ > 1:
                            print("NNLS stall detected... exiting")
                        exit_flag = 2
                        break
                elif tolerance_met:
                    if self.verbosity_ > 1:
                        print("Stall check: Residual near zero and tolerance met.")
                    exit_flag = 0
                    break

            mu = mat.T @ res
            if P_idx:
                mu[P_idx] = -np.inf
            if stalled_indices:
                mu[list(stalled_indices)] = -np.inf

            mumax = np.max(mu)
            imax = np.argmax(mu)

            tmp_mu_tol = mat.T @ rhs_halfgap
            mu_tol = 1.0e-15 * np.max(tmp_mu_tol) if tmp_mu_tol.size > 0 else 0.0

            if mumax < mu_tol:
                if stalled_indices:
                    if self.verbosity_ > 0:
                        print(f"Lagrange multiplier below threshold. Resetting "
                              f"{len(stalled_indices)} stalled indices.")
                    stalled_indices.clear()
                    mu = mat.T @ res
                    if P_idx:
                        mu[P_idx] = -np.inf
                    mumax = np.max(mu)
                    imax = np.argmax(mu)
                    if mumax < mu_tol:
                        if self.verbosity_ > 0:
                            print("Multipliers still below threshold after reset. "
                                  "Likely converged.")
                        exit_flag = 0
                        break
                else:
                    if self.verbosity_ > 0:
                        print("All Lagrange multipliers non-positive. Converged.")
                    exit_flag = 0
                    break

            if self.verbosity_ > 2:
                print(f"Adding index {imax} with multiplier {mumax:.6e}")
                sys.stdout.flush()

            P_set.add(imax)
            P_idx = sorted(list(P_set))
            n_glob = len(P_idx)
            stalled_now = False

            soln_nz_prev = np.copy(soln[P_idx])

            for iiter in range(self.n_inner_):
                n_total_inner_iter += 1
                A_P = mat[:, P_idx]

                try:
                    Q, R = sla.qr(A_P, mode='economic')
                    qt_b = Q.T @ rhs_avg
                    x_P_up = sla.solve_triangular(R, qt_b, check_finite=False)
                except np.linalg.LinAlgError:
                    if self.verbosity_ > 0:
                        print("Error: Linear algebra error in subproblem.")
                    exit_flag = 3
                    stalled_now = True
                    break

                min_x_P_up = np.min(x_P_up) if x_P_up.size > 0 else 0.0
                if min_x_P_up > -self.zero_tol_:
                    soln[P_idx] = np.maximum(x_P_up, 0)
                    if self.verbosity_ > 2:
                        print(f"  Inner iter {iiter}: Subproblem solution non-negative.")
                    break

                if self.verbosity_ > 2:
                    print(f"  Inner iter {iiter}: Pruning needed "
                          f"(min val: {min_x_P_up:.2e})")

                try:
                    new_idx_pos = P_idx.index(imax)
                    if x_P_up[new_idx_pos] <= self.zero_tol_:
                        if self.verbosity_ > 2:
                            print(f"  Stall detected: Last added index {imax} "
                                  "resulted in non-positive value.")
                        stalled_now = True
                        stalled_indices.add(imax)
                        break
                except ValueError:
                    if self.verbosity_ > 1:
                        print(f"  Warning: imax {imax} not found in P_idx {P_idx}.")

                alpha = 1.0e300
                for j in range(n_glob):
                    if x_P_up[j] <= self.zero_tol_:
                        denominator = soln_nz_prev[j] - x_P_up[j]
                        if abs(denominator) > 1e-15:
                            alpha = min(alpha, soln_nz_prev[j] / denominator)

                if alpha > 1.0e300:
                    alpha = 1.0

                soln[P_idx] = soln_nz_prev + alpha * (x_P_up - soln_nz_prev)
                indices_to_remove = {
                    P_idx[j] for j in range(n_glob) if soln[P_idx[j]] < self.zero_tol_
                }

                if not indices_to_remove:
                    min_pos_idx = -1
                    if len(P_idx) > 0:
                        min_pos_val = np.inf
                        for j in range(len(P_idx)):
                            if soln[P_idx[j]] < min_pos_val:
                                min_pos_val = soln[P_idx[j]]
                                min_pos_idx = P_idx[j]
                        if min_pos_idx != -1:
                            indices_to_remove = {min_pos_idx}
                            if self.verbosity_ > 1:
                                print(f"  Warning: Alpha calculation didn't force zero? "
                                      f"Forcing removal of index {min_pos_idx}.")

                if self.verbosity_ > 2:
                    print(f"  Alpha={alpha:.4f}, Removing indices: {indices_to_remove}")

                P_set.difference_update(indices_to_remove)
                P_idx = sorted(list(P_set))
                n_glob = len(P_idx)
                soln[list(indices_to_remove)] = 0.0

                if n_glob > 0:
                    soln_nz_prev = np.copy(soln[P_idx])
                else:
                    break

            if stalled_now:
                if self.verbosity_ > 2:
                    print(f"Reverting addition of stalled index {imax}")
                P_set.discard(imax)
                soln[imax] = 0.0
                continue

        if self.verbosity_ > 0:
            if exit_flag == 0:
                print("NNLS converged.")
            elif exit_flag == 1:
                print("NNLS terminated: Maximum outer iterations reached.")
            elif exit_flag == 2:
                print("NNLS terminated: Stalled (no progress).")
            elif exit_flag == 3:
                print("NNLS terminated: Other reason (e.g., subproblem failure, M ≤ N).")

        final_soln = np.zeros(n_tot)
        if P_set:
            P_idx = sorted(list(P_set))
            final_soln[P_idx] = np.maximum(soln[P_idx], 0)
        return final_soln, exit_flag