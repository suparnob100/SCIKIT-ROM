"""DEIM and S-OPT sampling for nonlinear reduced-order models.

TL;DR
-----
Build a nonlinear POD basis, select sampled DOFs with DEIM or S-OPT, and
construct the reduced gappy-POD operator used during online assembly.
"""

from __future__ import annotations

import numpy as np

from skrom.utils.reduced_basis.svd import svd_mode_selector


class deim:
    """Train a DEIM/gappy-POD operator from nonlinear snapshots.

    This class keeps the original scikit-rom public workflow intact:

    ``deim(mesh_or_basis, F_nl, V_sel).select_elems()``

    still returns ``(deim_mat, sampled_rows)`` and stores the selected element
    mask in ``self.xi``.  The internals are stricter about snapshot orientation,
    avoid duplicate samples, preserve S-OPT row order, and support optional
    oversampling without counting ``extra_modes`` twice.
    """

    def __init__(
        self,
        mesh,
        F_nl,
        V_sel,
        tol_f=1e-2,
        extra_modes=0,
        oversampling=0,
        free_dofs=None,
        candidate_rows=None,
        verbose=False,
    ):
        self.sampling_space = mesh
        self.mesh = getattr(mesh, "mesh", mesh)
        self.F_nl = np.asarray(F_nl, dtype=float)
        self.V = np.asarray(V_sel, dtype=float)
        self.tol_f = float(tol_f)
        self.extra_modes = _nonnegative_integer(extra_modes, "extra_modes")
        self.oversampling = _nonnegative_integer(
            oversampling,
            "oversampling",
        )
        self.free_dofs = _optional_index_array(free_dofs, "free_dofs")
        self.candidate_rows = _optional_index_array(
            candidate_rows,
            "candidate_rows",
        )
        self._row_map = None
        self._candidate_local_rows = None
        self.verbose = bool(verbose)
        self._snapshots = self._validated_snapshots()

    def _validated_snapshots(self):
        if self.F_nl.ndim != 2 or self.F_nl.size == 0:
            raise ValueError(
                "F_nl must be a non-empty 2D snapshot matrix."
            )
        if not np.all(np.isfinite(self.F_nl)):
            raise ValueError("F_nl must contain only finite values.")
        if self.V.ndim != 2 or not np.all(np.isfinite(self.V)):
            raise ValueError("V_sel must be a finite 2D basis matrix.")
        if not 0.0 <= self.tol_f < 1.0:
            raise ValueError("tol_f must lie in [0, 1).")

        dof_count = self.V.shape[0]
        if self.F_nl.shape[1] == dof_count:
            snapshots = self.F_nl
        elif self.F_nl.shape[0] == dof_count:
            snapshots = self.F_nl.T
        else:
            raise ValueError(
                "One axis of F_nl must match the number of rows in V_sel."
            )

        self._row_map = None
        candidate_rows = self.candidate_rows
        if self.free_dofs is not None:
            if np.unique(self.free_dofs).size != self.free_dofs.size:
                raise ValueError("free_dofs must not contain duplicate indices.")
            if self.free_dofs.size == dof_count:
                self._row_map = self.free_dofs
            elif candidate_rows is None:
                candidate_rows = self.free_dofs

        self._candidate_local_rows = self._local_candidate_rows(
            candidate_rows,
            dof_count,
        )
        return snapshots

    def _local_candidate_rows(self, candidate_rows, dof_count):
        if candidate_rows is None:
            return None
        rows = np.asarray(candidate_rows, dtype=np.intp).ravel()
        if rows.size == 0:
            raise ValueError("candidate_rows must not be empty.")
        if np.unique(rows).size != rows.size:
            raise ValueError("candidate_rows must not contain duplicate indices.")

        if self._row_map is None:
            if rows.max() >= dof_count:
                raise ValueError(
                    "candidate_rows must be valid row indices of V_sel."
                )
            return rows

        global_to_local = {
            int(global_row): local_row
            for local_row, global_row in enumerate(self._row_map)
        }
        if all(int(row) in global_to_local for row in rows):
            return np.asarray(
                [global_to_local[int(row)] for row in rows],
                dtype=np.intp,
            )
        if rows.max() < dof_count:
            return rows
        raise ValueError(
            "candidate_rows must be local rows or global rows present in "
            "free_dofs."
        )

    def select_elems(self, sopt=False, *, selection=None):
        """Select sampled DOFs and construct the reduced interpolation matrix."""
        strategy = _selection_name(selection, sopt)
        # nonlinear_basis, singular_values = _left_singular_vectors(
        #     self._snapshots
        # )
        # selected = _mode_count_from_tolerance(singular_values, self.tol_f)

        selected, nonlinear_basis = svd_mode_selector(self._snapshots,    # Input: mean-centered training snapshots
                                    tolerance=self.tol_f,    # Convergence criterion for mode selection
                                    modes=True)        # Return both number of modes and basis vectors


        n_modes = min(
            selected + self.extra_modes,
            nonlinear_basis.shape[1],
        )
        if n_modes <= 0:
            raise ValueError("At least one nonlinear POD mode is required.")

        self.U_fs = nonlinear_basis[:, :n_modes]
        self.singular_values = None #singular_values
        self.n_f_sel = n_modes
        n_samples = n_modes + self.oversampling
        if n_samples > self.U_fs.shape[0]:
            raise ValueError(
                f"Requested {n_samples} samples for only "
                f"{self.U_fs.shape[0]} available DOFs."
            )

        if strategy == "sopt":
            sampled_basis, local_rows, _ = self.sopt_red(
                self.U_fs,
                n_modes,
                num_samples=n_samples,
                candidate_rows=self._candidate_local_rows,
            )
        else:
            sampled_basis, local_rows = self.deim_red(
                self.U_fs,
                n_modes,
                num_samples=n_samples,
                candidate_rows=self._candidate_local_rows,
            )

        local_rows = np.asarray(local_rows, dtype=np.intp)
        global_rows = self._global_sample_indices(local_rows)
        self.xi = self._element_mask(global_rows)
        self.sampled_rows = global_rows
        self.local_sampled_rows = local_rows
        self.selection = strategy
        self.sampled_basis = sampled_basis
        self.interpolation_condition = float(np.linalg.cond(sampled_basis))
        self.deim_mat = self.V.T @ self.U_fs @ np.linalg.pinv(sampled_basis)

        if self.verbose:
            print(
                f"Selected {n_modes} nonlinear modes and {n_samples} "
                f"{strategy.upper()} samples."
            )
        return self.deim_mat, global_rows

    def _global_sample_indices(self, local_rows):
        if self._row_map is None:
            return local_rows.copy()
        return self._row_map[local_rows]

    def _element_mask(self, global_rows):
        if hasattr(self.sampling_space, "element_dofs"):
            element_dofs = np.asarray(
                self.sampling_space.element_dofs,
                dtype=np.intp,
            ).T
            n_full = int(self.sampling_space.N)
        elif hasattr(self.mesh, "t") and hasattr(self.mesh, "p"):
            n_full = int(self.mesh.p.shape[1])
            if self.free_dofs is None and self.V.shape[0] != n_full:
                raise ValueError(
                    "A mesh can map DEIM samples only for scalar nodal "
                    "spaces. Pass the scikit-fem Basis for higher-order, "
                    "vector, mixed, or compact free-DOF spaces."
                )
            element_dofs = np.asarray(self.mesh.t, dtype=np.intp).T
        else:
            raise TypeError(
                "DEIM training requires a scikit-fem Basis or mesh-like object."
            )

        if global_rows.size and global_rows.max() >= n_full:
            raise ValueError(
                "A sampled global DOF lies outside the supplied sampling space."
            )
        return np.any(np.isin(element_dofs, global_rows), axis=1).astype(float)

    @staticmethod
    def deim_red(
        f_basis,
        num_f_basis_vectors_used,
        *,
        num_samples=None,
        candidate_rows=None,
    ):
        """Select rows using original or oversampled greedy DEIM."""
        basis = _truncated_basis(f_basis, num_f_basis_vectors_used)
        n_dofs, n_modes = basis.shape
        n_samples = n_modes if num_samples is None else int(num_samples)
        candidates = _candidate_rows(candidate_rows, n_dofs)
        if not n_modes <= n_samples <= candidates.size:
            raise ValueError(
                "num_samples must be between the retained mode count and "
                "the candidate-row count."
            )

        sampled_rows = [
            int(candidates[np.argmax(np.abs(basis[candidates, 0]))])
        ]
        if n_samples == 1:
            return basis[sampled_rows, :], sampled_rows

        if n_modes == 1:
            order = np.argsort(-np.abs(basis[candidates, 0]), kind="stable")
            sampled_rows = candidates[order[:n_samples]].astype(int).tolist()
            return basis[sampled_rows, :], sampled_rows

        samples_per_mode = int(np.ceil((n_samples - 1) / (n_modes - 1)))
        candidate_mask = np.ones(n_dofs, dtype=bool)
        candidate_mask[:] = False
        candidate_mask[candidates] = True
        for mode in range(1, n_modes):
            for _ in range(samples_per_mode):
                previous = basis[:, :mode]
                sampled_previous = previous[sampled_rows, :]
                coefficients = np.linalg.pinv(sampled_previous) @ basis[
                    sampled_rows,
                    mode,
                ]
                residual = np.abs(
                    basis[:, mode] - previous @ coefficients
                )
                residual[~candidate_mask] = -np.inf
                residual[sampled_rows] = -np.inf
                sampled_rows.append(int(np.argmax(residual)))
                if len(sampled_rows) == n_samples:
                    return basis[sampled_rows, :], sampled_rows

        return basis[sampled_rows, :], sampled_rows

    @staticmethod
    def sopt_red(
        f_basis,
        num_f_basis_vectors_used,
        *,
        num_samples=None,
        candidate_rows=None,
    ):
        """Select rows by maximizing the S-OPT objective."""
        basis = _truncated_basis(f_basis, num_f_basis_vectors_used)
        Q, _ = np.linalg.qr(basis, mode="reduced")
        n_dofs, n_modes = Q.shape
        n_samples = n_modes if num_samples is None else int(num_samples)
        candidates = _candidate_rows(candidate_rows, n_dofs)
        if not n_modes <= n_samples <= candidates.size:
            raise ValueError(
                "num_samples must be between the retained mode count and "
                "the candidate-row count."
            )

        first = int(candidates[np.argmax(np.abs(Q[candidates, 0]))])
        selected = [first]
        remaining = candidates[candidates != first]

        for column in range(1, n_modes):
            chosen = np.asarray(selected, dtype=np.intp)
            A = Q[chosen, :column]
            gram_inverse = np.linalg.inv(A.T @ A)
            c = Q[chosen, column]
            g = gram_inverse @ (A.T @ c)
            column_norms = np.sum(A * A, axis=0)
            c_norm = float(c @ c)

            R = Q[remaining, :column]
            gamma = Q[remaining, column]
            B = R @ gram_inverse
            rtb = np.sum(R * B, axis=1)
            log_first = np.log(np.maximum(1.0 + rtb, 1e-300))
            log_first -= np.sum(
                np.log(np.maximum(column_norms[None, :] + R * R, 1e-300)),
                axis=1,
            )

            cta = c @ A
            scores = np.full(remaining.size, -np.inf)
            identity = np.eye(column)
            for candidate in range(remaining.size):
                denominator = 1.0 + rtb[candidate]
                total_norm = c_norm + gamma[candidate] ** 2
                if denominator <= 0.0 or total_norm <= 0.0:
                    continue
                projection = identity - np.outer(
                    B[candidate],
                    R[candidate],
                ) / denominator
                alpha = (
                    (cta + gamma[candidate] * R[candidate])
                    @ projection
                    @ (g + gamma[candidate] * B[candidate])
                )
                second = total_norm - alpha
                if second > 0.0:
                    scores[candidate] = (
                        log_first[candidate]
                        + np.log(second)
                        - np.log(total_norm)
                    )

            best_position = _best_score_position(
                scores,
                Q,
                selected,
                remaining,
                column + 1,
            )
            selected.append(int(remaining[best_position]))
            remaining = np.delete(remaining, best_position)

        if n_samples > n_modes:
            A = Q[selected, :]
            gram_inverse = np.linalg.pinv(A.T @ A)
            column_norms = np.sum(A * A, axis=0)

            for _ in range(n_samples - n_modes):
                R = Q[remaining, :]
                V = R @ gram_inverse
                leverage = np.sum(V * R, axis=1)
                scores = np.log(np.maximum(1.0 + leverage, 1e-300))
                scores -= np.sum(
                    np.log(
                        np.maximum(
                            column_norms[None, :] + R * R,
                            1e-300,
                        )
                    ),
                    axis=1,
                )
                best_position = _best_score_position(
                    scores,
                    Q,
                    selected,
                    remaining,
                    n_modes,
                )
                best_row = R[best_position]
                best_update = V[best_position]
                denominator = 1.0 + leverage[best_position]
                selected.append(int(remaining[best_position]))
                remaining = np.delete(remaining, best_position)
                gram_inverse -= np.outer(
                    best_update,
                    best_update,
                ) / denominator
                column_norms += best_row * best_row

        selected_mask = np.zeros(n_dofs, dtype=bool)
        selected_mask[selected] = True
        return basis[selected, :], selected, selected_mask


def _left_singular_vectors(snapshots):
    U, singular_values, _ = np.linalg.svd(snapshots.T, full_matrices=False)
    return U, singular_values


def _mode_count_from_tolerance(singular_values, tolerance):
    values = np.asarray(singular_values, dtype=float).reshape(-1)
    if values.size == 0:
        raise ValueError("At least one singular value is required.")
    total = float(np.sum(values * values))
    if total <= 0.0:
        return 1
    trailing = np.cumsum(values[::-1] ** 2)[::-1]
    error_after_k = np.concatenate((trailing[1:], np.array([0.0])))
    relative_error = np.sqrt(error_after_k / total)
    selected = np.flatnonzero(relative_error <= float(tolerance))
    return int(selected[0] + 1) if selected.size else values.size


def _nonnegative_integer(value, name):
    if not isinstance(value, (int, np.integer)) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer.")
    return int(value)


def _optional_index_array(value, name):
    if value is None:
        return None
    rows = np.asarray(value, dtype=np.intp).ravel()
    if rows.size == 0:
        raise ValueError(f"{name} must not be empty.")
    if rows.min() < 0:
        raise ValueError(f"{name} must contain non-negative indices.")
    return rows


def _candidate_rows(candidate_rows, row_count):
    if candidate_rows is None:
        return np.arange(row_count, dtype=np.intp)
    rows = np.asarray(candidate_rows, dtype=np.intp).ravel()
    if rows.size == 0:
        raise ValueError("candidate_rows must not be empty.")
    if rows.min() < 0 or rows.max() >= row_count:
        raise ValueError("candidate_rows contains an invalid basis row.")
    if np.unique(rows).size != rows.size:
        raise ValueError("candidate_rows must not contain duplicate indices.")
    return rows


def _selection_name(selection, legacy_sopt):
    if selection is None:
        return "sopt" if legacy_sopt else "deim"
    name = (
        str(selection)
        .strip()
        .lower()
        .replace("-", "")
        .replace("_", "")
        .replace(" ", "")
    )
    if name not in {"deim", "sopt"}:
        raise ValueError("selection must be 'deim' or 'sopt'.")
    if legacy_sopt and name != "sopt":
        raise ValueError("sopt=True conflicts with selection='deim'.")
    return name


def _truncated_basis(f_basis, requested_modes):
    basis = np.asarray(f_basis, dtype=float)
    if basis.ndim != 2 or basis.size == 0:
        raise ValueError("f_basis must be a non-empty two-dimensional array.")
    modes = min(int(requested_modes), basis.shape[1])
    if modes <= 0:
        raise ValueError("At least one basis vector is required.")
    return basis[:, :modes]


def _sopt_value(matrix):
    gram = matrix.T @ matrix
    sign, log_determinant = np.linalg.slogdet(gram)
    column_norms = np.linalg.norm(matrix, axis=0)
    if sign <= 0.0 or np.any(column_norms == 0.0):
        return -np.inf
    return (
        0.5 * log_determinant - np.sum(np.log(column_norms))
    ) / matrix.shape[1]


def _best_score_position(scores, Q, selected, remaining, column_count):
    if np.any(np.isfinite(scores)):
        return int(np.argmax(scores))
    exact_scores = np.array(
        [
            _sopt_value(Q[[*selected, int(candidate)], :column_count])
            for candidate in remaining
        ]
    )
    return int(np.argmax(exact_scores))


DEIM = deim

__all__ = ["DEIM", "deim"]
