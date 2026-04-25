"""
Discrete Empirical Interpolation Method (DEIM) for nonlinear ROM acceleration.

This module implements DEIM for reducing the dimension of nonlinear force terms in 
reduced-order models within finite element frameworks. DEIM enables efficient 
evaluation of nonlinear terms by:
- Computing empirical modes from nonlinear force snapshots via SVD
- Selecting optimal interpolation points using greedy algorithms
- Constructing projection matrices for fast nonlinear term approximation
- Mapping selected DOFs to element indicators for efficient assembly

**TL;DR**: Dramatically reduces computational cost of nonlinear ROM evaluation 
by approximating nonlinear terms using interpolation at carefully selected points,
achieving significant speedups while maintaining accuracy.

Author: Suparno Bhattacharyya
"""

import numpy as np
from skrom.fom.fem_utils import *  # Import FEM utility functions (e.g., element matrix computations)
from skrom.rom.rom_utils import *             # Import ROM utility functions (e.g., snapshot handling, SVD selectors)
from skrom.utils.reduced_basis.svd import *
from scipy.linalg import qr


class deim:
    """
    Discrete Empirical Interpolation Method for nonlinear ROM acceleration.

    **TL;DR**: Reduces computational cost of nonlinear terms in ROMs by ~1000x 
    through strategic sampling and interpolation, enabling real-time nonlinear 
    PDE solutions.

    The Discrete Empirical Interpolation Method (DEIM) addresses the computational 
    bottleneck in nonlinear reduced-order models where nonlinear terms must still 
    be evaluated at all degrees of freedom. DEIM constructs an efficient 
    approximation by:

    1. **Empirical Mode Analysis**: Computes dominant modes of nonlinear force 
       snapshots using SVD to capture the essential nonlinear behavior patterns.

    2. **Optimal Point Selection**: Uses a greedy algorithm to select interpolation 
       points that maximize information content while minimizing approximation error.

    3. **Projection Matrix Construction**: Builds a projection matrix that enables 
       fast reconstruction of full nonlinear terms from interpolated values.

    4. **Element Mapping**: Maps selected degrees of freedom to finite element 
       indicators for efficient sparse assembly operations.

    The method transforms the nonlinear term evaluation from O(n) to O(m) where 
    m << n, achieving dramatic computational savings essential for real-time 
    applications.

    Parameters
    ----------
    mesh : object
        Finite element mesh containing connectivity information and node data.
    F_nl : ndarray of shape (n_dofs, n_snapshots)  
        Snapshot matrix of nonlinear force evaluations at various parameter values.
        Each column represents the nonlinear force vector for one parameter instance.
    V_sel : ndarray of shape (n_dofs, n_modes)
        Reduced basis matrix from POD, used for solution space projection.
    tol_f : float, default=1e-2
        Tolerance for SVD mode selection based on singular value decay. Smaller 
        values retain more modes for higher accuracy.
    extra_modes : int, default=0
        Additional empirical modes to retain beyond those selected by tolerance
        criterion, useful for capturing marginal nonlinear effects.

    Attributes
    ----------
    U_fs : ndarray
        Truncated empirical basis matrix containing selected nonlinear modes.
    deim_mat : ndarray  
        DEIM projection matrix enabling fast nonlinear term reconstruction:
        F_nl_approx = V.T @ U_fs @ deim_mat @ F_nl_sampled
    xi : ndarray of int
        Binary element indicator vector marking which elements contain selected DOFs.
    n_f_sel : int
        Number of empirical modes selected for DEIM approximation.

    Notes
    -----
    DEIM is particularly effective for:
    - Nonlinear PDEs with localized nonlinear effects
    - Real-time control applications requiring fast ROM evaluation  
    - Problems where nonlinear term evaluation dominates computational cost
    - Systems with smooth nonlinear behavior amenable to low-rank approximation

    The method assumes the nonlinear terms can be well-approximated by a low-rank 
    representation, which is typically valid for many physical systems.

    References
    ----------
    .. [1] Chaturantabut, S. and Sorensen, D.C., 2010. Nonlinear model reduction 
           via discrete empirical interpolation method. SIAM journal on scientific 
           computing, 32(5), pp.2737-2764.

    Examples
    --------
    >>> # Generate nonlinear force snapshots
    >>> F_snapshots = compute_nonlinear_snapshots(problem, param_list)
    >>> # Create DEIM approximation
    >>> deim_obj = deim(mesh, F_snapshots, reduced_basis, tol_f=1e-3)
    >>> deim_matrix, sample_points = deim_obj.select_elems()
    >>> # Use in ROM: F_approx = V.T @ U @ deim_matrix @ F_sampled
    """

    def __init__(self, mesh, F_nl, V_sel, tol_f=1e-2, extra_modes=0):
        """
        Initialize DEIM with nonlinear snapshots and reduction parameters.

        Parameters
        ----------
        mesh : object
            Finite element mesh object containing node connectivity matrix `t` 
            and other geometric information needed for DOF-to-element mapping.
        F_nl : ndarray of shape (n_dofs, n_snapshots)
            Matrix of nonlinear force vector snapshots, where each column contains 
            the nonlinear force evaluation at one parameter instance.
        V_sel : ndarray of shape (n_dofs, n_modes)
            Selected reduced basis matrix, typically from POD truncation of 
            solution snapshots.
        tol_f : float, default=1e-2
            SVD tolerance for selecting empirical modes. Modes with normalized 
            singular values below this threshold are discarded.
        extra_modes : int, default=0
            Number of additional modes to retain beyond the tolerance-based 
            selection, providing extra approximation capacity.

        Notes
        -----
        The nonlinear snapshots F_nl should span the expected parameter range 
        and capture the full variety of nonlinear behaviors. Quality of DEIM 
        approximation depends critically on the representativeness of these snapshots.
        """
        self.mesh = mesh                          # Store finite element mesh data
        self.tol_f = tol_f                        # Tolerance for singular value thresholding
        self.V = V_sel                            # Reduced basis matrix 
        self.extra_modes = extra_modes            # Number of extra modes beyond tolerance selection
        self.F_nl = F_nl                          # Nonlinear force snapshot matrix

    def select_elems(self, sopt=False):
        """
        Select interpolation points and construct DEIM projection matrix.

        **TL;DR**: Core DEIM algorithm that identifies optimal sampling points 
        and builds the projection matrix for fast nonlinear term approximation.

        This method performs the complete DEIM setup:

        1. **SVD Analysis**: Decomposes nonlinear snapshots to identify dominant 
           empirical modes that capture essential nonlinear behavior patterns.

        2. **Mode Selection**: Applies tolerance-based truncation with optional 
           extra modes to balance accuracy and computational efficiency.

        3. **Point Selection**: Uses greedy DEIM algorithm to select interpolation 
           points that minimize approximation error in the empirical subspace.

        4. **Element Mapping**: Maps selected DOFs to finite element indicators 
           for efficient sparse matrix assembly during online evaluation.

        5. **Projection Construction**: Builds the DEIM projection matrix that 
           enables reconstruction: F_full ≈ U_fs @ pinv(U_fs[selected_rows, :]) @ F_sampled

        Returns
        -------
        deim_mat : ndarray of shape (n_modes, n_selected_points)
            DEIM projection matrix for reconstructing full nonlinear terms from 
            sampled values. Used as: F_approx = V.T @ U_fs @ deim_mat @ F_sampled
        sampled_rows : list of int
            Indices of degrees of freedom selected as DEIM interpolation points.
            These are the only DOFs where nonlinear terms need evaluation.

        Notes
        -----
        The projection matrix construction uses the Moore-Penrose pseudoinverse 
        to ensure numerical stability even when the empirical basis is not 
        perfectly conditioned.

        The element mapping (self.xi) enables efficient assembly by identifying 
        which finite elements contribute to the selected DOFs, allowing sparse 
        matrix operations during online evaluation.

        Examples
        --------
        >>> deim_obj = deim(mesh, F_snapshots, basis_matrix)
        >>> proj_matrix, points = deim_obj.select_elems()
        >>> print(f"Selected {len(points)} points from {F_snapshots.shape[0]} DOFs")
        >>> print(f"Reduction ratio: {F_snapshots.shape[0]/len(points):.1f}x")
        """
        # Determine the number of modes to retain using an SVD-based selector
        n_f_sel, U_f = svd_mode_selector(self.F_nl, self.tol_f)
        n_f_sel += self.extra_modes  # Include any additional modes specified
        print(f"Selected modes: {n_f_sel}")
        
        # Truncate the full basis to only include the selected modes
        U_fs = U_f[:, :n_f_sel]
        
        # Use the DEIM sampling strategy to select interpolation points
        
        if sopt:
            f_basis_sampled, sampled_rows, _ = self.sopt_red(U_f, n_f_sel)
        else:
            f_basis_sampled, sampled_rows = self.deim_red(U_f, n_f_sel)

        # Map selected DOF indices to element indicators
        # Elements containing any selected DOF are marked for assembly
        deim_dof = np.any(np.isin(self.mesh.t.T, sampled_rows), axis=1)

        # Convert the selected DOFs to a binary element indicator vector
        self.xi = deim_dof.astype(int)
        
        # Build the DEIM projection matrix using pseudoinverse for numerical stability
        # This matrix enables: F_full ≈ U_fs @ deim_mat @ F_sampled
        self.deim_mat = self.V.T @ U_fs @ np.linalg.pinv(f_basis_sampled)
        
        # Store additional attributes for potential later use
        self.U_fs = U_fs
        self.n_f_sel = n_f_sel

        return self.deim_mat, sampled_rows

    def deim_red(self, f_basis, num_f_basis_vectors_used):
        """
        Execute greedy DEIM algorithm for optimal interpolation point selection.

        **TL;DR**: Implements the core greedy algorithm that iteratively selects 
        interpolation points to minimize approximation error in the empirical subspace.

        The DEIM greedy algorithm works by:

        1. **Initial Selection**: Chooses the DOF with maximum absolute value in 
           the first empirical mode as the starting interpolation point.

        2. **Iterative Refinement**: For each subsequent mode, solves for optimal 
           interpolation coefficients using previously selected points, then 
           selects the DOF with maximum residual as the next point.

        3. **Residual Minimization**: Each new point is chosen to minimize the 
           approximation error when reconstructing the current empirical mode 
           from previously selected points.

        This greedy strategy ensures that interpolation points capture maximum 
        information content while maintaining numerical stability through 
        well-conditioned interpolation matrices.

        Parameters
        ----------
        f_basis : ndarray of shape (n_dofs, n_modes)
            Empirical basis matrix from SVD of nonlinear force snapshots.
            Each column represents one empirical mode.
        num_f_basis_vectors_used : int
            Number of empirical modes to use for interpolation point selection.
            Cannot exceed the total number of available modes.

        Returns
        -------
        f_basis_sampled : ndarray of shape (num_modes, num_modes)
            Square matrix containing rows of the empirical basis corresponding 
            to selected interpolation points. Used to construct the projection matrix.
        sampled_rows : list of int
            Ordered list of DOF indices selected as interpolation points.
            The order reflects the greedy selection sequence.

        Notes
        -----
        The algorithm ensures that the interpolation matrix f_basis_sampled 
        remains well-conditioned by construction, as each new point is chosen 
        to maximize the residual norm.

        For problems with strong locality in nonlinear effects, DEIM typically 
        selects points near regions of highest nonlinear activity, leading to 
        physically intuitive sampling patterns.

        The computational complexity is O(m³) where m is the number of modes, 
        making it efficient even for moderately large empirical subspaces.

        Examples
        --------
        >>> # Empirical basis from SVD  
        >>> U, s, Vt = np.linalg.svd(F_snapshots, full_matrices=False)
        >>> sampled_basis, indices = deim_obj.deim_red(U, n_modes=10)
        >>> print(f"Condition number: {np.linalg.cond(sampled_basis):.2e}")
        """
        # Ensure we don't request more modes than available
        num_basis_vectors = min(num_f_basis_vectors_used, f_basis.shape[1])
        basis_size = f_basis.shape[0]
        
        # Initialize matrix to store sampled rows from the empirical basis
        f_basis_sampled = np.zeros((num_basis_vectors, num_basis_vectors))
        
        # Track selected DOF indices and sampling status
        sampled_rows = []
        is_sampled = np.zeros(basis_size, dtype=bool)

        # Initial selection: DOF with maximum absolute value in first mode
        f_bv_max_global_row = np.argmax(np.abs(f_basis[:, 0]))
        sampled_rows.append(f_bv_max_global_row)
        is_sampled[f_bv_max_global_row] = True

        # Store the first row in the sampled basis matrix
        f_basis_sampled[0, :] = f_basis[f_bv_max_global_row, :num_basis_vectors]

        # Greedy selection for remaining modes
        for i in range(1, num_basis_vectors):
            # Solve interpolation problem: find coefficients c such that
            # f_basis_sampled[:i, :i] @ c ≈ f_basis_sampled[:i, i]
            c = np.linalg.solve(f_basis_sampled[:i, :i], f_basis_sampled[:i, i])
            
            # Compute residual when approximating i-th mode using first i-1 modes
            # at selected interpolation points
            r_val = np.abs(f_basis[:, i] - np.dot(f_basis[:, :i], c))
            
            # Select DOF with maximum residual as next interpolation point
            f_bv_max_global_row = np.argmax(r_val)
            
            # Update selection lists and sampling status
            sampled_rows.append(f_bv_max_global_row)
            is_sampled[f_bv_max_global_row] = True

            # Add new row to sampled basis matrix
            f_basis_sampled[i, :] = f_basis[f_bv_max_global_row, :num_basis_vectors]

        return f_basis_sampled, sampled_rows
    
  
    
    def sopt_red(self, f_basis, num_f_basis_vectors_used):
        """
        S-OPT sampling: selects rows maximising the S-optimality measure
        S(Z^T Q) which jointly maximises column orthogonality and determinant
        of the information matrix (Z^T Q)^T (Z^T Q).

        Uses efficient Sherman-Morrison rank-1 updates to avoid recomputing
        determinants at every candidate evaluation.

        Phase 1 (j < n_f): row + column expansion   — Slide 12/14 formula
        Phase 2 (j >= n_f): row-only append (oversampling) — Slide 13/15 formula

        Parameters
        ----------
        f_basis : ndarray (N, n_modes_total)
            Full empirical basis from SVD of nonlinear snapshots.
        num_f_basis_vectors_used : int
            Number of basis vectors (modes) to use.

        Returns
        -------
        f_sampled_row : ndarray
            Rows of f_basis at the selected indices.
        Z_indices : list of int
            Selected row indices.
        Z_boolean : ndarray of bool
            Boolean mask of selected rows.

        References
        ----------
        Shin & Xiu, SIAM J. Sci. Comput. 38 (2016), pp. A385-A411.
        Lauzon et al., SIAM J. Sci. Comput. 46 (2024), arXiv:2203.16494.
        """
        num_sampling_points = num_f_basis_vectors_used + self.extra_modes
        n_f = min(num_f_basis_vectors_used, f_basis.shape[1])
        f_basis_trunc = f_basis[:, :n_f]

        # Orthogonal basis via QR
        Q, _ = np.linalg.qr(f_basis_trunc)
        N = Q.shape[0]

        # ── first index: largest |entry| in first column ──
        i_star = int(np.argmax(np.abs(Q[:, 0])))
        Z_indices = [i_star]
        remaining = list(range(N))
        remaining.remove(i_star)

        # ════════════════════════════════════════════════════════════════
        # Phase 1:  j = 1 .. n_f-1   (row + column expansion)
        #
        #   Ã = [A c; r^T γ]  =>
        #   S(Ã)^{2(m+1)} = det(A^T A) · (1+r^Tb)/∏(||Ae_k||²+r_k²)
        #                   · (c^Tc+γ²-α)/(c^Tc+γ²)
        #   b=(A^TA)^{-1}r, g=(A^TA)^{-1}A^Tc,
        #   α=(c^TA+γr^T)(I-(1+r^Tb)^{-1}br^T)(g+γb)
        # ════════════════════════════════════════════════════════════════
        for j in range(1, n_f):
            sel = np.array(Z_indices)
            m = len(sel)  # == j (square matrix A is m×m)

            A_mat = Q[sel, :j]                          # (m, j)
            G_inv = np.linalg.inv(A_mat.T @ A_mat)      # (j, j)
            c_vec = Q[sel, j]                            # (m,)
            g_vec = G_inv @ (A_mat.T @ c_vec)            # (j,)
            col_sq = np.sum(A_mat ** 2, axis=0)          # (j,)
            ctc = float(c_vec @ c_vec)

            # Vectorised over candidates
            cand = np.array(remaining)
            R = Q[cand, :j]                              # (|cand|, j)
            Gamma = Q[cand, j]                           # (|cand|,)

            B = R @ G_inv                                # (|cand|, j)
            rtb = np.sum(R * B, axis=1)                  # (|cand|,)

            # factor1 = (1+r^Tb) / ∏(col_sq + r_k²)
            log_f1 = (np.log(np.maximum(1.0 + rtb, 1e-300))
                      - np.sum(np.log(col_sq[None, :] + R ** 2), axis=1))

            ctA = c_vec @ A_mat                          # (j,)
            denom2 = ctc + Gamma ** 2                    # (|cand|,)

            scores = np.full(len(cand), -np.inf)
            for ci in range(len(cand)):
                if 1.0 + rtb[ci] <= 0:
                    continue
                ri = R[ci]
                bi = B[ci]
                gam = Gamma[ci]

                vec1 = ctA + gam * ri
                proj = np.eye(j) - np.outer(bi, ri) / (1.0 + rtb[ci])
                alpha = vec1 @ proj @ (g_vec + gam * bi)

                f2_num = ctc + gam ** 2 - alpha
                if f2_num <= 0 or denom2[ci] <= 0:
                    continue
                scores[ci] = log_f1[ci] + np.log(f2_num) - np.log(denom2[ci])

            best = cand[int(np.argmax(scores))]
            Z_indices.append(best)
            remaining.remove(best)

        # ════════════════════════════════════════════════════════════════
        # Phase 2:  oversampling (row-only append, j >= n_f)
        #
        #   Ã = [A; r^T]  =>
        #   S(Ã)^{2p} = det(A^TA) · (1+r^T(A^TA)^{-1}r) / ∏(||Ae_k||²+r_k²)
        #
        #   det(A^TA) is constant across candidates, so maximise:
        #       (1 + r^T G_inv r) / ∏(col_sq_k + r_k²)
        #
        #   G_inv updated via Sherman-Morrison after each selection.
        # ════════════════════════════════════════════════════════════════
        if num_sampling_points > n_f:
            sel = np.array(Z_indices)
            A_mat = Q[sel, :]                            # (|sel|, n_f)
            G = A_mat.T @ A_mat
            try:
                G_inv = np.linalg.inv(G)
            except np.linalg.LinAlgError:
                G_inv = np.linalg.pinv(G)
            col_sq = np.sum(A_mat ** 2, axis=0)          # (n_f,)

            for _ in range(num_sampling_points - n_f):
                cand = np.array(remaining)
                R = Q[cand, :]                           # (|cand|, n_f)

                # Vectorised score
                V = R @ G_inv                            # (|cand|, n_f)
                rtGr = np.sum(V * R, axis=1)
                log_num = np.log(np.maximum(1.0 + rtGr, 1e-300))
                log_den = np.sum(np.log(col_sq[None, :] + R ** 2), axis=1)
                scores = log_num - log_den

                bi = int(np.argmax(scores))
                best = cand[bi]
                r_best = R[bi]
                v_best = V[bi]

                Z_indices.append(best)
                remaining.remove(best)

                # Sherman-Morrison update
                G_inv -= np.outer(v_best, v_best) / (1.0 + rtGr[bi])
                col_sq += r_best ** 2

        # ── pack outputs ──
        Z_boolean = np.zeros(N, dtype=bool)
        Z_boolean[Z_indices] = True
        f_sampled_row = f_basis_trunc[Z_boolean, :]

        print(f"{Z_indices=}")
        return f_sampled_row, Z_indices, Z_boolean
    
    
