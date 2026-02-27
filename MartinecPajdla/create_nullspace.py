"""
create_nullspace - Create null-space for scene with perspective camera
Converted from MATLAB to Python
"""

import numpy as np
import time
from typing import Tuple, Dict
from MartinecPajdla.Utils import Utils


def create_nullspace(M: np.ndarray, 
                    depths: np.ndarray, 
                    central: int, 
                    opt: dict = None) -> Tuple[np.ndarray, Dict]:
    """
    Create null-space for scene with perspective camera.
    
    Args:
        M: Measurement matrix (3m x n)
        depths: Binary matrix indicating scaled elements (m x n)
                1 = scaled by known depth, 0 = not scaled
        central: Central image index (0 for sequence mode)
        opt: Options dict with:
            - 'trial_coef': Coefficient for number of trials (default 1)
            - 'threshold': Threshold for null-space (default 0.01)
            - 'verbose': Print progress (default False)
    
    Returns:
        nullspace: Null-space matrix
        result: Dict with 'used', 'failed', 'tried' counts
    """
    
    # Default options
    if opt is None:
        opt = {'trial_coef': 1, 'threshold': 0.01, 'verbose': False}
    if 'trial_coef' not in opt:
        opt['trial_coef'] = 1
    if 'threshold' not in opt:
        opt['threshold'] = 0.01
    if 'verbose' not in opt:
        opt['verbose'] = False
    
    # Check which points are valid (non-NaN)
    I = ~np.isnan(M[0::3, :])  # Check x-coordinates
    m, n = I.shape
    
    show_mod = 10
    use_maxtuples = 0
    
    if opt['verbose']:
        print(f'Used 4-tuples (.={show_mod}): ', end='', flush=True)
        start_time = time.time()
    
    # Determine which columns are scaled
    if central > 0:
        cols_scaled = np.zeros(n, dtype=int)
        cols_scaled[I[central, :]] = 1
    else:
        cols_scaled = np.array([])
    
    num_trials = round(opt['trial_coef'] * n)
    
    # Allocate memory for nullspace
    if opt['verbose']:
        print('(Allocating memory...', end='', flush=True)
    
    nullspace = np.zeros((M.shape[0], num_trials))
    width = 0
    
    if opt['verbose']:
        print(')', end='', flush=True)
    
    # Progress tracking
    tenth = 0.1
    result = {'used': 0, 'failed': 0, 'tried': 0}
    
    # Main loop: try random 4-tuples
    for i in range(num_trials):
        # Initialize for this trial
        cols = np.arange(n)
        rows = np.arange(m)
        cols_chosen = []
        failed = False
        
        if central > 0:
            scaled_ensured = False
        else:
            scaled_ensured = True  # No scale control in sequence mode
        
        # Choose 4 columns
        for t in range(4):
            # Randomly choose one column
            c, cols = random_element(cols)
            cols_chosen.append(c)
            
            # Keep only rows valid for this column
            rows = np.intersect1d(rows, np.where(I[:, c])[0])
            
            # Cut useless rows/cols before choosing next
            if t < 3:
                rows, cols, scaled_ensured = cut_useless(
                    I, cols_scaled, cols_chosen, rows, cols, 
                    4 - t - 1, scaled_ensured
                )
            
            if len(rows) == 0:
                failed = True
                break
        
        if not failed:
            # Use the 4-tuple
            cols_chosen = np.array(cols_chosen)
            d = depths[np.ix_(rows, cols_chosen)]
            
            # Build submatrix
            rowsbig = Utils.k2i(rows, step=3)
            submatrix_parts = []
            
            for j in range(len(cols_chosen)):
                col_data = M[rowsbig, cols_chosen[j]].reshape(-1, 1)
                depth_col = d[:, j].reshape(-1, 1)
                spread = Utils.spread_depths_col(col_data, depth_col)
                submatrix_parts.append(spread)
            
            submatrix = np.hstack(submatrix_parts)
            
            # Debug output
            if opt['verbose'] and submatrix.shape[0] <= submatrix.shape[1]:
                print('-', end='', flush=True)
            
            # Compute null-space
            subnull = nulleps(submatrix, opt['threshold'])
            
            # Check if valid null-space
            if subnull.shape[1] > 0 and (
                use_maxtuples or 
                submatrix.shape[0] == submatrix.shape[1] + subnull.shape[1]
            ):
                # Create null-space vector
                nulltemp = np.zeros((M.shape[0], subnull.shape[1]))
                nulltemp[rowsbig, :] = subnull
                
                # Expand nullspace array if needed
                if width + nulltemp.shape[1] > nullspace.shape[1]:
                    if opt['verbose']:
                        print('(Allocating memory...)', end='', flush=True)

                    mean_added = width / (i + 1) if i > 0 else 1
                    estimated_extra = round(mean_added * (num_trials - i - 1))
                    # CRITICAL: must guarantee at least nulltemp.shape[1] new cols
                    # so the write immediately below never overflows
                    needed = width + nulltemp.shape[1] - nullspace.shape[1]
                    extra_cols = max(estimated_extra, needed)
                    nullspace = np.hstack([nullspace, np.zeros((M.shape[0], extra_cols))])
                
                # Add to nullspace
                nullspace[:, width:width + nulltemp.shape[1]] = nulltemp
                width += nulltemp.shape[1]
                result['used'] += 1
                
                if result['used'] % show_mod == 0 and opt['verbose']:
                    print('.', end='', flush=True)
        else:
            result['failed'] += 1
        
        # Progress reporting
        if (i + 1) / num_trials > 0.1 * tenth:
            if opt['verbose']:
                print(f'{int(tenth * 10)}%', end='', flush=True)
            if tenth < 1:
                tenth = 0
            tenth += 1
    
    # Cut unused memory
    if nullspace.shape[1] > width:
        if opt['verbose']:
            print('(Cutting unused memory...', end='', flush=True)
        nullspace = nullspace[:, :width]
        if opt['verbose']:
            print(')', end='', flush=True)
    
    if opt['verbose']:
        elapsed = time.time() - start_time
        print(f' ({elapsed:.3f} sec)')
    
    result['tried'] = num_trials
    
    return nullspace, result


def cut_useless(I: np.ndarray,
               cols_scaled: np.ndarray,
               cols_chosen: list,
               rows: np.ndarray,
               cols: np.ndarray,
               demanded: int,
               scaled_ensured: bool) -> Tuple[np.ndarray, np.ndarray, bool]:
    """
    Cut useless rows and columns based on data availability.
    
    Args:
        I: Validity matrix (m x n)
        cols_scaled: Which columns are scaled
        cols_chosen: Already chosen columns
        rows: Current row candidates
        cols: Current column candidates
        demanded: Number of additional columns needed
        scaled_ensured: Whether scaling requirement is met
    
    Returns:
        rows: Filtered rows
        cols: Filtered columns
        scaled_ensured: Updated scaling status
    """
    
    if not scaled_ensured:
        # Determine requirements
        if len(rows) == 2:
            demanded_scaled = 3
            demanded_rows = 2
        else:
            demanded_scaled = 2
            demanded_rows = 3
        
        # Count scaled columns already chosen
        if len(cols_scaled) > 0:
            cols_scaled_chosen = np.sum(cols_scaled[cols_chosen] > 0)
        else:
            cols_scaled_chosen = 0
        
        # If all remaining must be scaled, filter columns
        if demanded == demanded_scaled - cols_scaled_chosen:
            if len(cols_scaled) > 0:
                cols = np.intersect1d(cols, np.where(cols_scaled > 0)[0])
            scaled_ensured = True
    else:
        demanded_rows = 2
    
    # Filter columns: keep those with enough valid rows
    if len(cols) > 0:
        col_sums = np.sum(I[np.ix_(rows, cols)], axis=0)
        valid_cols = cols[col_sums >= demanded_rows]
        cols = valid_cols
    
    # Filter rows: keep those with enough valid columns
    if len(rows) > 0 and len(cols) > 0:
        row_sums = np.sum(I[np.ix_(rows, cols)], axis=1)
        valid_rows = rows[row_sums >= demanded]
        rows = valid_rows
    
    return rows, cols, scaled_ensured


def random_element(arr: np.ndarray) -> Tuple[int, np.ndarray]:
    """
    Take a random element from array.
    
    Args:
        arr: Input array
    
    Returns:
        element: Random element
        rest: Array without that element
    """
    idx = np.random.randint(0, len(arr))
    element = arr[idx]
    rest = np.delete(arr, idx)
    return element, rest


def nulleps(M: np.ndarray, tol: float) -> np.ndarray:
    """
    Find null-space of M with tolerance.
    Returns vectors with singular values <= tol.
    
    Args:
        M: Input matrix
        tol: Threshold for singular values
    
    Returns:
        N: Null-space basis (vectors with small singular values)
    """
    
    if M.size == 0:
        return np.zeros((M.shape[0], 0))
    
    try:
        # Compute SVD
        u, s, vt = np.linalg.svd(M, full_matrices=True)
        
        # Count significant singular values
        sigsvs = np.sum(s > tol)
        
        # Return left singular vectors beyond significant ones
        N = u[:, sigsvs:]
        
    except np.linalg.LinAlgError:
        # SVD failed - return empty null-space
        N = np.zeros((M.shape[0], 0))
    
    return N


if __name__ == "__main__":
    # Test create_nullspace function
    print("=" * 60)
    print("create_nullspace - Null-Space Creation Test")
    print("=" * 60)
    
    np.random.seed(42)
    
    # Create test data
    m_images = 5
    n_points = 20
    
    # Measurement matrix (3m x n)
    M = np.random.randn(3 * m_images, n_points) * 100
    
    # Add some missing data
    missing = np.random.rand(3 * m_images, n_points) < 0.1
    M[missing] = np.nan
    
    # Depths matrix (m x n) - binary
    depths = np.random.randint(0, 2, (m_images, n_points))
    
    print(f"\nTest data:")
    print(f"  Images: {m_images}")
    print(f"  Points: {n_points}")
    print(f"  M shape: {M.shape}")
    print(f"  Depths shape: {depths.shape}")
    print(f"  Missing entries: {np.sum(missing)}")
    
    # Test 1: Sequence mode
    print("\n" + "-" * 60)
    print("TEST 1: Sequence Mode")
    print("-" * 60)
    
    opt = {
        'trial_coef': 0.5,
        'threshold': 0.01,
        'verbose': True
    }
    
    nullspace_seq, result_seq = create_nullspace(M, depths, central=0, opt=opt)
    
    print(f"\nResults:")
    print(f"  Nullspace shape: {nullspace_seq.shape}")
    print(f"  Trials: {result_seq['tried']}")
    print(f"  Used: {result_seq['used']}")
    print(f"  Failed: {result_seq['failed']}")
    
    # Test 2: Central mode
    print("\n" + "-" * 60)
    print("TEST 2: Central Image Mode (central=2)")
    print("-" * 60)
    
    opt_quiet = {
        'trial_coef': 0.3,
        'threshold': 0.01,
        'verbose': False
    }
    
    nullspace_cent, result_cent = create_nullspace(M, depths, central=2, opt=opt_quiet)
    
    print(f"Results:")
    print(f"  Nullspace shape: {nullspace_cent.shape}")
    print(f"  Trials: {result_cent['tried']}")
    print(f"  Used: {result_cent['used']}")
    print(f"  Failed: {result_cent['failed']}")
    print(f"  Success rate: {result_cent['used']/result_cent['tried']*100:.1f}%")
    
    # Test 3: Helper functions
    print("\n" + "-" * 60)
    print("TEST 3: Helper Functions")
    print("-" * 60)
    
    # Test random_element
    arr = np.array([1, 2, 3, 4, 5])
    elem, rest = random_element(arr)
    print(f"random_element([1,2,3,4,5]): elem={elem}, rest={rest}")
    
    # Test nulleps
    test_matrix = np.array([[1, 0, 0],
                            [0, 0.001, 0],
                            [0, 0, 0.0001]])
    null = nulleps(test_matrix, tol=0.01)
    print(f"\nnulleps with tol=0.01:")
    print(f"  Input singular values: ~[1, 0.001, 0.0001]")
    print(f"  Nullspace dimension: {null.shape[1]}")
    print(f"  (Should be 2, corresponding to small singular values)")
    
    # Test cut_useless
    I_test = np.array([[1, 1, 0, 1],
                       [1, 1, 1, 0],
                       [0, 1, 1, 1]], dtype=bool)
    cols_scaled_test = np.array([1, 0, 1, 0])
    rows_test = np.array([0, 1, 2])
    cols_test = np.array([0, 1, 2, 3])
    cols_chosen_test = [0]
    
    rows_cut, cols_cut, _ = cut_useless(
        I_test, cols_scaled_test, cols_chosen_test,
        rows_test, cols_test, demanded=2, scaled_ensured=True
    )
    print(f"\ncut_useless test:")
    print(f"  Filtered rows: {rows_cut}")
    print(f"  Filtered cols: {cols_cut}")
    
    print("\n" + "=" * 60)
    print("✓ All tests completed")
    print("=" * 60)
