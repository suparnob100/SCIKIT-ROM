import numpy as np
from skrom.fom.fem_utils import *  # Import FEM utility functions (e.g., element matrix computations)
from skrom.rom.rom_utils import *             # Import ROM utility functions (e.g., snapshot handling, SVD selectors)
from skrom.utils.reduced_basis.svd import *
from scipy.linalg import qr

class deim:
    """
    Class to perform the Discrete Empirical Interpolation Method (DEIM) for 
    reducing the dimension of nonlinear force terms in a reduced order model 
    (ROM) within a finite element framework.
    
    [Author: Suparno Bhattacharyya]
    """

    def __init__(self, mesh, F_nl, V_sel, tol_f=1e-2, extra_modes=0):
        """
        Constructor to initialize DEIM parameters and input data.
        
        Parameters:
        -----------
        d : object
            An instance containing the FEM model data (e.g., node connectivity, equation IDs).
        F_nl : ndarray
            Snapshot matrix containing nonlinear force evaluations for various training samples.
        train_mask : array_like (boolean or index array)
            Mask to select the training snapshots from sol_snapshots.
        param_list : list
            List of parameter values corresponding to each training snapshot.
        V : ndarray
            Reduced basis matrix (projection basis for the full-order solution).
        tol_f : float, optional
            Tolerance for selecting the number of singular value modes from the SVD (default is 1e-2).
        extra_modes : int, optional
            Additional modes to include beyond those selected by the tolerance criterion.
        extra_samples : int, optional
            Additional sampling points for oversampling in the S-optimal strategy.
        """
        self.mesh = mesh                          # Store problem_class (e.g., mesh, eqn IDs, etc.)
        self.tol_f = tol_f                        # Tolerance for singular value thresholding
        self.V = V_sel                            # Reduced basis matrix (could be further masked if needed)
        self.extra_modes = extra_modes            # Number of extra modes to be added to the selected basis modes
        self.F_nl = F_nl

    def select_elems(self):
        """
        Select the elements (or rows) for DEIM reduction based on the computed snapshot basis.
        This method:
          1. Performs SVD on the nonlinear force snapshots to determine dominant modes.
          2. Uses an S-optimal sampling strategy (or DEIM, if uncommented) to select rows.
          3. Maps the selected DOFs back to element indicators.
          4. Constructs the DEIM projection matrix.
        """
        # Determine the number of modes to retain using an SVD-based selector.
        n_f_sel, U_f = svd_mode_selector(self.F_nl, self.tol_f)
        n_f_sel += self.extra_modes  # Include any additional modes specified
        print(f"Selected modes:{n_f_sel}")
        
        # Truncate the full basis to only include the selected modes.
        U_fs = U_f[:, :n_f_sel]
        
        # Use the DEIM sampling strategy to select rows.
        # Alternatively use S-optimal sampling strategy.

        f_basis_sampled, sampled_rows = self.deim_red(U_f, n_f_sel)


        # Retrieve the global equation IDs for the nodes corresponding to the provided mask.
        deim_dof = np.any(np.isin(self.mesh.t.T, sampled_rows), axis = 1)

        # Convert the selected DOFs to an element indicator vector.
        self.xi = deim_dof.astype(int)
        
        # Build the DEIM projection matrix using the reduced basis and the pseudo-inverse of the sampled basis.
        self.deim_mat = self.V.T @ U_fs @ np.linalg.pinv(f_basis_sampled)
        
        # Store additional attributes for potential later use.
        self.U_fs = U_fs
        self.n_f_sel = n_f_sel

        return self.deim_mat, sampled_rows



    def deim_red(self, f_basis, num_f_basis_vectors_used):
        """
        Perform the standard Discrete Empirical Interpolation Method (DEIM) to select rows 
        that capture the dominant features of the nonlinear basis.
        
        Parameters:
        -----------
        f_basis : ndarray
            Basis matrix derived from the nonlinear force snapshots.
        num_f_basis_vectors_used : int
            Number of basis vectors (modes) to use for the reduction.
        
        Returns:
        --------
        f_basis_sampled : ndarray
            Matrix of the selected rows from the original basis.
        sampled_rows : list
            List of row indices that were selected.
        is_sampled : ndarray
            Boolean array indicating which rows of the original basis are selected.
        """
        # Determine the effective number of basis vectors to use (cannot exceed total available).
        num_basis_vectors = min(num_f_basis_vectors_used, f_basis.shape[1])
        basis_size = f_basis.shape[0]
        
        # Initialize a matrix to store the sampled rows (each row will contain a subset of the basis)
        f_basis_sampled = np.zeros((num_basis_vectors, num_basis_vectors))
        
        # List to store the indices of the rows selected by DEIM.
        sampled_rows = []
        
        # Boolean array to track which rows have been sampled.
        is_sampled = np.zeros(basis_size, dtype=bool)

        # Select the index corresponding to the maximum absolute value in the first basis vector.
        f_bv_max_global_row = np.argmax(np.abs(f_basis[:, 0]))
        sampled_rows.append(f_bv_max_global_row)
        is_sampled[f_bv_max_global_row] = True

        # Store the corresponding row from the first basis vector into the sampled matrix.
        f_basis_sampled[0, :] = f_basis[f_bv_max_global_row, :num_basis_vectors]

        # Iteratively select subsequent rows for each additional basis vector.
        for i in range(1, num_basis_vectors):
            # Solve for the interpolation coefficients that best approximate the i-th basis vector 
            # using the previously selected rows.
            c = np.linalg.solve(f_basis_sampled[:i, :i], f_basis_sampled[:i, i])
            
            # Compute the residual of the approximation for the i-th basis vector.
            r_val = np.abs(f_basis[:, i] - np.dot(f_basis[:, :i], c))
            
            # Select the row with the maximum residual as the next sampling index.
            f_bv_max_global_row = np.argmax(r_val)
            
            # Update the list of sampled rows and the corresponding boolean indicator.
            sampled_rows.append(f_bv_max_global_row)
            is_sampled[f_bv_max_global_row] = True

            # Update the sampled basis matrix with the newly selected row.
            f_basis_sampled[i, :] = f_basis[f_bv_max_global_row, :num_basis_vectors]

        # Return the sampled basis matrix, the list of selected row indices, and the boolean mask.
        return f_basis_sampled, sampled_rows



    def deim_dof_to_elem(self, deim_dof):
        """
        Map the selected degrees-of-freedom (DOFs) to an element indicator vector.
        Each element (or cell) is flagged if any of its nonzero node equation IDs 
        appear in the DEIM-selected DOFs.
        
        Parameters:
        -----------
        deim_dof : list
            List of DOF indices selected by the DEIM or S-optimal procedure.
        
        Returns:
        --------
        x : ndarray
            A binary vector indicating for each element whether it contains any
            of the selected DOFs (1 if yes, 0 if no).
        """
        # Retrieve the list of nonzero equation IDs for each global node associated with elements.
        glob_node_nonzero_eqnId = self.cls.glob_node_nonzero_eqnId

        # Initialize an array to mark elements; default is 0 (not selected).
        x = np.zeros(len(glob_node_nonzero_eqnId))

        # Loop over each element and mark it if any of its node equation IDs are in the selected DOFs.
        for iel in range(len(glob_node_nonzero_eqnId)):
            # Check if any equation ID in the current element is among the DEIM-selected DOFs.
            bool_array = np.isin(glob_node_nonzero_eqnId[iel], deim_dof)
            if np.any(bool_array):                
                x[iel] = 1  # Mark the element as selected

        return x