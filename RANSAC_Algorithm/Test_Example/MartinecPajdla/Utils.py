import numpy as np
from typing import List, Tuple, Union, Optional
import warnings


class Utils:
    """
    A utility class containing mathematical functions converted from MATLAB.
    This class provides functionality for:
    - Combinatorial operations (comb, combnext)
    - Random number generation (random_int, diff_rand_ints)
    - Distance calculations (dist, eucl_dist, eucl_dist_only)
    - Coordinate transformations (normalize_cut, normalize_mp, p2e, hom, k2i)
    - Normalization (normu, normP, normx)
    - Matrix operations (spread_depths_col, subseq_longest)
    - Radial distortion (raddist_apply, raddist_deriv)

    """

    @staticmethod
    def comb(n: int, k: int) -> int:
        """
        Returns combination number n over k (n choose k).

        Args:
            n: Total number of items
            k: Number of items to choose

        Returns:
            Combination number C(n,k)
        """
        if k > n or k < 0:
            return 0
        if k == 0 or k == n:
            return 1

        r = 1
        for i in range(1, k + 1):
            r = r * (n - i + 1) // i
        return r

    @staticmethod
    def combnext(n: int, k: int, com: np.ndarray) -> np.ndarray:
        """
        Returns the next combination in order of shifting the least left
        number to the right.

        Args:
            n: Total number of items
            k: Number of items to choose
            com: Current combination

        Returns:
            Next combination as numpy array
        """
        next_com = com.copy()
        move = k - 1  # Convert to 0-based indexing
        moved = False

        while not moved:
            if next_com[move] < n - k + move:
                next_com[move] += 1
                for i in range(move + 1, k):
                    next_com[i] = next_com[move] + i - move
                moved = True
            else:
                if move > 0:
                    move -= 1
                else:
                    raise ValueError('Error: this code should have never been called')

        return next_com

    @staticmethod
    def diff_rand_ints(ints: List[int], n: int, from_val: int, to_val: int) -> List[int]:
        """
        Add n random integers in some scope to a given vector.
        All resulting numbers should be different.

        Args:
            ints: Existing list of integers
            n: Number of new random integers to add
            from_val: Lower bound (inclusive)
            to_val: Upper bound (inclusive)

        Returns:
            List with n new unique random integers added
        """
        if (to_val + 1 - from_val) < n + len(ints):
            raise ValueError(f'Not enough room for {n} random numbers from {from_val} to {to_val}')

        result = ints.copy()

        for i in range(n):
            x = Utils.random_int(from_val, to_val - len(result))
            oldx = from_val - 1

            while oldx < x:
                temp = x
                # Count how many existing numbers fall in the range (oldx, x]
                count = sum(1 for num in result if oldx < num <= x)
                x = x + count
                oldx = temp

            result.append(x)

        return result

    @staticmethod
    def dist(M1: np.ndarray, M2: np.ndarray, metric: int = 2) -> float:
        """
        Return distance between image points in homogeneous coordinates in specified metric.

        Args:
            M1: First matrix of points
            M2: Second matrix of points
            metric: 1 = euclidean distance, 2 = standard deviation of coordinates

        Returns:
            Distance according to specified metric
        """
        if metric == 1:
            # Euclidean distance metric
            I = (~np.isnan(M1[::3, :])) & (~np.isnan(M2[::3, :]))
            d = Utils.eucl_dist(M1, M2, I) / np.sum(I)
        elif metric == 2:
            # Standard deviation metric
            D = Utils.normalize_cut(M1) - Utils.normalize_cut(M2)
            # Get non-NaN indices
            i = np.where(~np.isnan(D[::2]))[0]
            if len(i) > 0:
                coords = np.concatenate([D[2 * i], D[2 * i + 1]])
                d = np.std(coords)
            else:
                d = 0
        else:
            raise ValueError('dist: unknown metric')

        return d

    @staticmethod
    def eucl_dist(M0: np.ndarray, M: np.ndarray, I: Optional[np.ndarray] = None) -> float:
        """
        Return Euclidean norm of the difference between two scenes.
        Note: Perspective cameras are assumed.

        Args:
            M0: Reference matrix
            M: Comparison matrix
            I: Optional mask for valid points

        Returns:
            Euclidean distance
        """
        if I is None:
            I = (~np.isnan(M0[::3, :])) & (~np.isnan(M[::3, :]))

        return Utils.eucl_dist_only(Utils.normalize_cut(M0), Utils.normalize_cut(M), I)[0]

    @staticmethod
    def eucl_dist_only(M0: np.ndarray, M: np.ndarray, I: Optional[np.ndarray] = None, step: int = 2) -> Tuple[
        float, np.ndarray]:
        """
        Return Euclidean norm of the difference between two scenes.

        Args:
            M0: Reference matrix
            M: Comparison matrix
            I: Optional mask for valid points
            step: Step size (2 for inhomogeneous coordinates)

        Returns:
            Tuple of (euclidean_norm, distances_array)
        """
        if I is None:
            I = (~np.isnan(M0[::2, :])) & (~np.isnan(M[::2, :]))

        if I.size > 0:
            m = M.shape[0] // step
            if I.shape[0] != m and M.size > 0:
                warnings.warn(f'Warning: eucl_dist_only: the height of I is bad, it should be equal to {m}')

        # I has shape (m, n) where m = M.shape[0] // step.
        # np.where(I.flatten()) gives flat ROW-MAJOR indices into (m, n).
        # We must NOT pass these to k2i() which would treat them as image indices.
        # Instead convert flat indices to (image, point) pairs directly.
        img_idx, pt_idx = np.where(I)
        if len(img_idx) == 0:
            return 0.0, np.array([])

        n_cols = M.shape[1]
        B = np.zeros((step, len(img_idx)))
        for s in range(step):
            rows = step * img_idx + s   # correct triplet/pair row in M
            B[s, :] = (M0[rows, pt_idx] - M[rows, pt_idx]) ** 2

        distances = np.sqrt(np.sum(B, axis=0))
        e_norm = np.sum(distances)

        return e_norm, distances

    @staticmethod
    def k2i(k: Union[int, np.ndarray, List], step: int = 3) -> np.ndarray:
        """
        Compute indices of matrix rows corresponding to views k with some step.

        Args:
            k: View indices (scalar or array)
            step: Step size (default 3)

        Returns:
            Array of row indices
        """
        k = np.atleast_1d(k).flatten()

        # Create indices: [1:step] + step*(k-1) for each k
        indices = []
        for ki in k:
            base_indices = np.arange(step) + step * ki
            indices.extend(base_indices)

        return np.array(indices)

    @staticmethod
    def normalize_cut(M: np.ndarray, I: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Normalizes homogeneous coordinates and cuts the last coordinate
        (which then equals 1).

        Args:
            M: Input matrix with homogeneous coordinates
            I: Optional mask for valid points

        Returns:
            Normalized matrix with last coordinate removed
        """
        m = M.shape[0] // 3

        if I is None:
            Mnorm = Utils.normalize_mp(M)
        else:
            Mnorm = Utils.normalize_mp(M, I)

        # Keep only x and y coordinates (remove z coordinates)
        rows_to_keep = []
        for i in range(m):
            rows_to_keep.extend([3 * i, 3 * i + 1])  # Keep x and y, skip z

        return Mnorm[rows_to_keep, :]

    @staticmethod
    def normalize_mp(M: np.ndarray, I: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Normalizes M by dividing each point by its homogeneous coordinate
        (these coordinates equal to ones afterwards).

        Args:
            M: Input matrix (3m x n) with homogeneous image coords
            I: Optional (m x n) mask for valid points (unused, kept for API compatibility)

        Returns:
            Normalized matrix (each 3-row triplet divided by its z/w coordinate)
        """
        m = M.shape[0] // 3
        eps = np.finfo(float).eps

        Mnorm = M.copy().astype(float)

        # For each camera k, divide the full (x,y,z) triplet by the z coordinate
        for k in range(m):
            z = M[3*k + 2, :]                          # z-coords for camera k
            valid = (~np.isnan(z)) & (np.abs(z) > eps)
            if np.any(valid):
                Mnorm[3*k:3*k+3, valid] = M[3*k:3*k+3, valid] / z[valid]
            # For degenerate (z≈0) points: leave as-is (matches MATLAB behaviour)

        return Mnorm

    @staticmethod
    def normP(P: np.ndarray) -> np.ndarray:
        """
        Normalize joint camera matrix so that norm(P(k2i(k),:),'fro') = 1 for each k.

        Args:
            P: Joint camera matrix of size (3*m, n) where m is number of cameras

        Returns:
            Normalized camera matrix where each 3-row block has Frobenius norm = 1
        """
        P = P.copy().astype(float)
        m = P.shape[0] // 3  # Number of cameras

        for k in range(m):
            rows = Utils.k2i(k, step=3)
            Pk = P[rows, :]
            frobenius_norm = np.sqrt(np.sum(Pk * Pk))
            if frobenius_norm > 0:
                P[rows, :] = Pk / frobenius_norm

        return P

    @staticmethod
    def normx(x: np.ndarray) -> np.ndarray:
        """
        Normalize MxN matrix so that the norm of each column is 1.

        Args:
            x: Input matrix of size (M, N)

        Returns:
            Normalized matrix where each column has unit norm
        """
        if x.size == 0:
            return x

        x = x.copy().astype(float)
        col_norms = np.sqrt(np.sum(x * x, axis=0))
        # Avoid division by zero
        col_norms[col_norms == 0] = 1
        x = x / col_norms

        return x

    @staticmethod
    def spidx(I: np.ndarray, J: np.ndarray, V: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate sparse matrix indices for a submatrix block.

        Creates all combinations (Cartesian product) of row indices I and column
        indices J, useful for constructing sparse matrices with scipy.sparse.

        Args:
            I: Row indices (1D array)
            J: Column indices (1D array)
            V: Values matrix of size (len(I), len(J)) to be flattened

        Returns:
            Tuple of (i, j, v) where:
            - i: Row indices for sparse matrix (flattened)
            - j: Column indices for sparse matrix (flattened)
            - v: Values (flattened)

        Example:
            I = [0, 1], J = [2, 3, 4]
            i = [0, 1, 0, 1, 0, 1]  (rows repeated for each column)
            j = [2, 2, 3, 3, 4, 4]  (each column repeated for each row)

        Usage with scipy:
            from scipy.sparse import coo_matrix
            i, j, v = Utils.spidx(I, J, V)
            sparse_matrix = coo_matrix((v, (i, j)), shape=(m, n))
        """
        I = np.atleast_1d(I).flatten()
        J = np.atleast_1d(J).flatten()

        # Create outer product grids (Cartesian product of indices)
        # i = I' * ones(1, len(J)) -> each column is I
        i = np.outer(I, np.ones(len(J)))
        # j = ones(len(I), 1) * J -> each row is J
        j = np.outer(np.ones(len(I)), J)

        # Flatten column-wise (Fortran order) to match MATLAB's (:) behavior
        i = i.flatten('F')
        j = j.flatten('F')
        v = np.asarray(V).flatten('F')

        return i, j, v

    @staticmethod
    def p2e(x_: np.ndarray) -> np.ndarray:
        """
        Projective to euclidean coordinates.
        Computes euclidean coordinates by dividing all rows but last by the last row.

        Args:
            x_: Projective coordinates of size (dim+1, N)

        Returns:
            Euclidean coordinates of size (dim, N)
        """
        if x_.size == 0:
            return np.array([])

        dim = x_.shape[0] - 1
        if dim <= 0:
            return np.array([])

        # Divide first dim rows by last row
        last_row = x_[dim, :]
        x = x_[:dim, :] / last_row

        return x

    @staticmethod
    def hom(x: np.ndarray) -> np.ndarray:
        """
        Euclidean to homogeneous (projective) coordinates.
        Adds a row of ones to the bottom.
        
        This is the inverse operation of p2e().

        Args:
            x: Euclidean coordinates (dim x N) or (dim,)

        Returns:
            Homogeneous coordinates (dim+1 x N) or (dim+1,)
        """
        if x.size == 0:
            return x

        # Add row of ones
        if x.ndim == 1:
            x_hom = np.append(x, 1)
        else:
            ones = np.ones((1, x.shape[1]))
            x_hom = np.vstack([x, ones])

        return x_hom

    @staticmethod
    def random_int(from_val: int, to_val: int) -> int:
        """
        Returns random integer in specified range.

        Args:
            from_val: Lower bound (inclusive)
            to_val: Upper bound (inclusive)

        Returns:
            Random integer in range [from_val, to_val]
        """
        return int(np.floor(from_val + (1 + to_val - from_val) * np.random.rand()))

    @staticmethod
    def normu(u: np.ndarray) -> np.ndarray:
        """
        Normalize image points to be used for LS (Least Squares) estimation.
        Finds normalization matrix A so that mean(A*u) = 0 and mean(||A*u||) = sqrt(2).
        Based on Hartley: In Defense of 8-point Algorithm, ICCV'95.

        Args:
            u: Image points of size (2,N) or (3,N) - homogeneous or Euclidean coordinates

        Returns:
            Normalization matrix A of size (3,3)
            Returns empty array if degenerate configuration detected

        Process:
            1. Convert to Euclidean coordinates if input is homogeneous
            2. Calculate mean of all points and center them around origin
            3. Calculate average distance from origin
            4. Scale so average distance equals sqrt(2)
            5. Build transformation matrix combining translation and scaling
        """
        # Step 1: Convert to Euclidean coordinates if input is homogeneous (3xN)
        if u.shape[0] == 3:
            u = Utils.p2e(u)

        # Step 2: Calculate mean of points and center them
        m = np.mean(u, axis=1, keepdims=True)  # Mean of each row (x and y coordinates)
        u_centered = u - m  # Center points around origin

        # Step 3: Calculate distances from origin for each point
        distances = np.sqrt(np.sum(u_centered * u_centered, axis=0))  # ||u|| for each point

        # Step 4: Calculate average distance and scaling factor
        r = np.mean(distances) / np.sqrt(2)  # Scale factor to make mean distance = sqrt(2)

        # Step 5: Handle degenerate case (all points at same location)
        if r == 0:
            return np.array([])  # Degenerate configuration

        # Step 6: Build normalization matrix A
        # A combines scaling (1/r) and translation (-m/r)
        A = np.diag([1 / r, 1 / r, 1])  # Scaling matrix
        A[0:2, 2] = -m.flatten() / r  # Translation component

        return A

    @staticmethod
    def spread_depths_col(Mdepthcol: np.ndarray, depthsIcol: np.ndarray) -> np.ndarray:
        """
        Spreads a depth column to a submatrix with some zeros.
        Used for organizing depth information in multi-view geometry.

        Args:
            Mdepthcol: Column vector with depth values
            depthsIcol: Indicator column (non-zero where depths are known)

        Returns:
            Submatrix with depths spread according to known/unknown pattern

        Process:
            1. Find positions where depths are known (non-zero indicators)
            2. Place ALL known depths in the FIRST column
            3. Place each unknown depth in subsequent columns
            4. Use k2i() to get proper row indices for each depth
        """
        m = depthsIcol.shape[0]
        n = 0  # Column counter

        # Initialize output submatrix
        total_cols = m  # Maximum possible columns needed
        submatrix = np.zeros((m * 3, total_cols))  # Assuming 3D points (k2i with step=3)

        # Step 1: Find known depths (non-zero indicators)
        known_depths = np.where(depthsIcol.flatten() != 0)[0]

        # Step 2: Process ALL known depths into ONE column
        if len(known_depths) > 0:
            rows = Utils.k2i(known_depths, step=3)  # Get row indices for ALL known depths
            submatrix[rows, n] = Mdepthcol.flatten()[rows]  # Place all in one column
            n += 1

        # Step 3: Process each unknown depth in separate columns
        unknown_depths = np.setdiff1d(np.arange(m), known_depths)
        for depth_idx in unknown_depths:
            rows = Utils.k2i(depth_idx, step=3)  # Get row indices for this depth
            submatrix[rows, n] = Mdepthcol.flatten()[rows]  # Place depth values
            n += 1

        # Return only the columns we actually used
        return submatrix[:, :n]

    @staticmethod
    def subseq_longest(I: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Find the longest continuous subsequences in columns of matrix I.
        Returns starting position and length of longest continuous subsequence
        of True/non-zero values in each column.

        Args:
            I: Binary/indicator matrix of size (m, n)

        Returns:
            Tuple of (b, len) where:
            - b: Starting positions of longest subsequences for each column
            - len: Lengths of longest subsequences for each column

        Process:
            1. For each column, scan through rows
            2. Count consecutive True/non-zero values
            3. Track longest sequence found and its starting position
            4. Return starting positions and lengths for all columns
        """
        m, n = I.shape

        # Initialize output arrays
        b = np.zeros(n, dtype=int)  # Starting positions
        lengths = np.zeros(n, dtype=int)  # Sequence lengths

        # Step 1: Process each column
        for p in range(n):
            # Step 2: Initialize sequence tracking for this column
            seq = np.zeros(m, dtype=int)  # Track sequence lengths at each position
            current_seq_start = 0  # Current sequence starting position

            # Step 3: Scan through rows in this column
            for i in range(m):
                if I[i, p]:  # If this position has a True/non-zero value
                    # Continue current sequence
                    seq[current_seq_start] += 1
                else:
                    # Break current sequence, start new one at next position
                    current_seq_start = i + 1

            # Step 4: Find longest sequence in this column
            lengths[p] = np.max(seq)  # Length of longest sequence

            # Step 5: Find starting position of longest sequence
            best_positions = np.where(seq == lengths[p])[0]  # All positions with max length
            b[p] = best_positions[0]  # Take first occurrence (like MATLAB)

        return b, lengths

    @staticmethod
    def raddist_apply(q: np.ndarray, u0: np.ndarray, kappa: float) -> np.ndarray:
        """
        Apply radial distortion to points.
        
        Model: q_dist = u0 + (1 + κ·r²)·(q - u0)
        
        Args:
            q: Points (2 x N) or (2,)
            u0: Distortion center (2,)
            kappa: Radial distortion coefficient
        
        Returns:
            q_dist: Distorted points (same shape as q)
        """
        if q.ndim == 1:
            q = q.reshape(2, 1)
        
        # Center points
        u0 = u0.reshape(2, 1)
        q_centered = q - u0
        
        # Compute radius squared
        r2 = np.sum(q_centered ** 2, axis=0, keepdims=True)
        
        # Apply radial distortion: q_dist = u0 + (1 + kappa*r²) * (q - u0)
        distortion_factor = 1 + kappa * r2
        q_dist = u0 + distortion_factor * q_centered
        
        return q_dist

    @staticmethod
    def raddist_deriv(u: np.ndarray, u0: np.ndarray, kappa: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute derivatives of radial distortion.
        
        Args:
            u: Undistorted point (2,)
            u0: Distortion center (2,)
            kappa: Radial distortion coefficient
        
        Returns:
            dqdu: Jacobian w.r.t. undistorted point (2 x 2)
            dqdu0: Jacobian w.r.t. distortion center (2 x 2)
            dqdkappa: Jacobian w.r.t. kappa (2 x 1)
        """
        u = u.reshape(2)
        u0 = u0.reshape(2)
        
        # Centered point
        du = u - u0
        
        # Radius squared
        r2 = np.sum(du ** 2)
        
        # Distortion factor
        f = 1 + kappa * r2
        
        # dq/du
        # q = u0 + f*(u - u0)
        # dq/du = f*I + kappa * 2*(u-u0)⊗(u-u0)
        dqdu = f * np.eye(2) + 2 * kappa * np.outer(du, du)
        
        # dq/du0
        # q = u0 + f*(u - u0)
        # dq/du0 = I - f*I - kappa * 2*(u-u0)⊗(u-u0)
        dqdu0 = np.eye(2) - f * np.eye(2) - 2 * kappa * np.outer(du, du)
        
        # dq/dkappa
        # q = u0 + (1 + kappa*r²)*(u - u0)
        # dq/dkappa = r² * (u - u0)
        dqdkappa = (r2 * du).reshape(2, 1)
        
        return dqdu, dqdu0, dqdkappa


if __name__ == "__main__":

    # Examples:
    # Combinatorial
    # Operations:
    # Calculate "10 choose 3"
    result = Utils.comb(10, 3)
    print(f"C(10,3) = {result}")  # Output: 120

    # Generate first combination manually for choosing 3 items from 5
    first_comb = np.array([0, 1, 2])
    next_comb = Utils.combnext(5, 3, first_comb)
    print(f"Next combination: {next_comb}")  # Output: [0 1 3]

    # Random
    # Number
    # Generation:
    # Generate random integer between 1 and 10
    rand_num = Utils.random_int(1, 10)
    print(f"Random number: {rand_num}")

    # Add 3 unique random numbers to existing list
    existing_numbers = [2, 7]
    new_numbers = Utils.diff_rand_ints(existing_numbers, 3, 1, 15)
    print(f"Extended list: {new_numbers}")  # e.g., [2, 7, 1, 9, 14]

    # Coordinate
    # Transformations:
    # Convert projective to Euclidean coordinates
    projective_coords = np.array([[6], [8], [2]])  # Point (6,8) in homogeneous coords
    euclidean_coords = Utils.p2e(projective_coords)
    print(f"Euclidean coordinates: {euclidean_coords}")  # Output: [[3], [4]]

    # Generate matrix row indices for views
    view_indices = Utils.k2i([0, 2], step=3)
    print(f"Row indices: {view_indices}")  # Output: [0 1 2 6 7 8]

    # Distance
    # Calculations:
    # Create two simple 2D point sets in homogeneous coordinates
    M1 = np.array([[1, 3], [2, 4], [1, 1]])  # Points (1,2) and (3,4)
    M2 = np.array([[2, 4], [3, 5], [1, 1]])  # Points (2,3) and (4,5)

    # Calculate distance using different metrics
    dist_euclidean = Utils.dist(M1, M2, metric=1)
    dist_std = Utils.dist(M1, M2, metric=2)
    print(f"Euclidean distance: {dist_euclidean:.3f}")
    print(f"Standard deviation distance: {dist_std:.3f}")

    # Image
    # Point
    # Normalization:
    # Normalize image points for least squares estimation
    points = np.array([[1, 3, 5], [2, 4, 6]])  # 2D points
    norm_matrix = Utils.normu(points)
    print(f"Normalization matrix:\\n{norm_matrix}")

    # Matrix
    # Operations:
    # Find longest continuous subsequences
    I = np.array([[1, 0, 1], [1, 1, 1], [0, 1, 0], [1, 1, 1]])
    start_pos, lengths = Utils.subseq_longest(I)
    print(f"Starting positions: {start_pos}, Lengths: {lengths}")

    # Homogeneous
    # Coordinate
    # Normalization:
    # Create points with different homogeneous coordinates
    M = np.array([[2, 6], [4, 8], [2, 2]])  # Points (1,2) and (3,4) scaled by 2

    # Normalize by homogeneous coordinates
    M_normalized = Utils.normalize_mp(M)
    print(f"Normalized matrix:\\n{M_normalized}")

    # Normalize and cut last coordinate
    M_cut = Utils.normalize_cut(M)
    print(f"Normalized and cut:\\n{M_cut}")  # Should be [[1, 3], [2, 4]]

    # Complete
    # Workflow
    # Example:
    # Generate all combinations of choosing 2 points from 4
    n_points, k_choose = 4, 2
    combinations = []

    # Start with first combination manually
    current_comb = np.array([0, 1])  # First combination
    combinations.append(current_comb.copy())

    # Generate all combinations
    try:
        while True:
            current_comb = Utils.combnext(n_points, k_choose, current_comb)
            combinations.append(current_comb.copy())
    except ValueError:
        pass  # No more combinations

    print(f"All combinations: {combinations}")
    # Output: [[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]]