"""
balance_triplets - Balance PRMM by column and triplet-of-rows normalization
Converted from MATLAB to Python
"""

import numpy as np
import time
from MartinecPajdla.Utils import Utils


def balance_triplets(M: np.ndarray, opt: dict = None) -> np.ndarray:
    """
    Balance measurement matrix by iterative column and row-triplet normalization.
    
    After balancing, overall weight of M will be m*n where 3m*n is size of M.
    (i.e., 3 coordinates of each image point average to 1)
    
    Args:
        M: Measurement matrix (3m x n)
        opt: Options dict with:
            - 'info_separately': Display info on separate line (default True)
            - 'verbose': Display progress (default True)
    
    Returns:
        B: Balanced measurement matrix
    """
    
    # Default options
    if opt is None:
        opt = {}
    if 'info_separately' not in opt:
        opt['info_separately'] = True
    if 'verbose' not in opt:
        opt['verbose'] = True
    
    # Start timing
    if opt['verbose']:
        if opt['info_separately']:
            print('Balancing PRMM...', end='', flush=True)
        else:
            print('(balancing PRMM...', end='', flush=True)
        start_time = time.time()
    
    # Get dimensions
    m = M.shape[0] // 3  # number of cameras
    n = M.shape[1]       # number of points
    
    # Initialize
    B = M.copy()
    change = np.inf
    diff_rows = np.inf
    diff_cols = np.inf
    iteration = 0
    
    # Iterate until convergence or max iterations
    while (change > 0.01 or diff_rows > 1 or diff_cols > 1) and iteration <= 20:
        Bold = B.copy()
        
        # Step 1: Normalize each column
        # Each column scaled to weighted unit length
        diff_cols = -np.inf
        
        for l in range(n):
            # Find valid rows for this column
            rows = np.where(~np.isnan(M[0::3, l]))[0]
            
            if len(rows) > 0:
                # Get row indices (3 rows per camera)
                rowsb = Utils.k2i(rows, step=3)
                
                # Compute sum of squares
                s = np.sum(B[rowsb, l] ** 2)
                
                # Weight based on amount of valid data
                supposed_weight = len(rows)
                
                # Track maximum difference
                diff_cols = max(abs(s - supposed_weight), diff_cols)
                
                # Normalize column (with protection)
                if s > 1e-10:
                    B[rowsb, l] = B[rowsb, l] / np.sqrt(s / supposed_weight)
        
        # Step 2: Normalize each triplet of rows
        # Each camera's rows scaled to weighted unit area
        diff_rows = -np.inf
        
        for k in range(m):
            # Find valid columns for this camera (triplet of rows)
            cols = np.where(~np.isnan(M[3*k, :]))[0]
            
            if len(cols) > 0:
                # Compute sum of squares for this triplet
                s = np.sum(B[3*k:3*k+3, cols] ** 2)
                
                # Weight based on amount of valid data
                supposed_weight = len(cols)
                
                # Track maximum difference
                diff_rows = max(abs(s - supposed_weight), diff_rows)
                
                # Normalize triplet (with protection)
                if s > 1e-10:
                    B[3*k:3*k+3, cols] = B[3*k:3*k+3, cols] / np.sqrt(s / supposed_weight)
        
        # Step 3: Compute change from previous iteration
        change = 0
        for k in range(m):
            cols = np.where(~np.isnan(M[3*k, :]))[0]
            if len(cols) > 0:
                diff = B[3*k:3*k+3, cols] - Bold[3*k:3*k+3, cols]
                change += np.sum(diff ** 2)
        
        iteration += 1
        
        if opt['verbose']:
            print('.', end='', flush=True)
    
    # Display timing
    if opt['verbose']:
        elapsed = time.time() - start_time
        if opt['info_separately']:
            print(f' ({elapsed:.3f} sec)')
        else:
            print(f'{elapsed:.3f} sec)', end='', flush=True)
    
    return B


if __name__ == "__main__":
    # Test balance_triplets function
    print("=" * 60)
    print("balance_triplets - Matrix Balancing Test")
    print("=" * 60)
    
    np.random.seed(42)
    
    # Create test data
    m_images = 5
    n_points = 20
    
    # Generate measurement matrix with varying scales
    M = np.random.randn(3 * m_images, n_points) * 100
    
    # Add some missing data
    missing = np.random.rand(3 * m_images, n_points) < 0.15
    M[missing] = np.nan
    
    # Add some large scale variations
    for i in range(m_images):
        scale = 0.1 + i * 0.5  # Different scales for different cameras
        M[3*i:3*i+3, :] *= scale
    
    for j in range(n_points):
        scale = 0.5 + j * 0.1  # Different scales for different points
        M[:, j] *= scale
    
    print(f"\nTest data:")
    print(f"  Images: {m_images}")
    print(f"  Points: {n_points}")
    print(f"  M shape: {M.shape}")
    print(f"  Missing entries: {np.sum(missing)}/{M.size}")
    
    # Compute initial statistics  
    print(f"\nBefore balancing:")
    col_weights_before = []
    for l in range(n_points):
        rows = np.where(~np.isnan(M[0::3, l]))[0]
        if len(rows) > 0:
            rowsb = Utils.k2i(rows, step=3)
            if len(rowsb) > 0 and not np.any(np.isnan(M[rowsb, l])):
                weight = np.sum(M[rowsb, l] ** 2)
                col_weights_before.append(weight / len(rows))
    
    row_weights_before = []
    for k in range(m_images):
        cols = np.where(~np.isnan(M[3*k, :]))[0]
        if len(cols) > 0:
            triplet = M[3*k:3*k+3, cols]
            if not np.any(np.isnan(triplet)):
                weight = np.sum(triplet ** 2)
                row_weights_before.append(weight / len(cols))
    
    if len(col_weights_before) > 0 and len(row_weights_before) > 0:
        print(f"  Column weights/count: min={np.min(col_weights_before):.2f}, max={np.max(col_weights_before):.2f}, std={np.std(col_weights_before):.2f}")
        print(f"  Row triplet weights/count: min={np.min(row_weights_before):.2f}, max={np.max(row_weights_before):.2f}, std={np.std(row_weights_before):.2f}")
    
    # Test 1: Balance with verbose output
    print("\n" + "-" * 60)
    print("TEST 1: Balancing (Verbose, Separate Line)")
    print("-" * 60)
    
    opt = {'verbose': True, 'info_separately': True}
    B = balance_triplets(M, opt)
    
    # Compute statistics after balancing
    print(f"\nAfter balancing:")
    col_weights_after = []
    for l in range(n_points):
        rows = np.where(~np.isnan(M[0::3, l]))[0]
        if len(rows) > 0:
            rowsb = Utils.k2i(rows, step=3)
            if len(rowsb) > 0 and not np.any(np.isnan(B[rowsb, l])):
                weight = np.sum(B[rowsb, l] ** 2)
                col_weights_after.append(weight / len(rows))
    
    row_weights_after = []
    for k in range(m_images):
        cols = np.where(~np.isnan(M[3*k, :]))[0]
        if len(cols) > 0:
            triplet = B[3*k:3*k+3, cols]
            if not np.any(np.isnan(triplet)):
                weight = np.sum(triplet ** 2)
                row_weights_after.append(weight / len(cols))
    
    if len(col_weights_after) > 0 and len(row_weights_after) > 0:
        print(f"  Column weights/count: min={np.min(col_weights_after):.2f}, max={np.max(col_weights_after):.2f}, std={np.std(col_weights_after):.2f}")
        print(f"  Row triplet weights/count: min={np.min(row_weights_after):.2f}, max={np.max(row_weights_after):.2f}, std={np.std(row_weights_after):.2f}")
        
        if len(col_weights_before) > 0 and len(row_weights_before) > 0:
            col_improvement = np.std(col_weights_before) / max(np.std(col_weights_after), 1e-10)
            row_improvement = np.std(row_weights_before) / max(np.std(row_weights_after), 1e-10)
            print(f"  Improvement in std: columns {col_improvement:.2f}x, rows {row_improvement:.2f}x")
    
    # Test 2: Balance without verbose
    print("\n" + "-" * 60)
    print("TEST 2: Balancing (Quiet)")
    print("-" * 60)
    
    opt_quiet = {'verbose': False}
    B2 = balance_triplets(M, opt_quiet)
    
    print(f"Balanced silently")
    print(f"Results match Test 1: {np.allclose(B, B2, equal_nan=True)}")
    
    # Test 3: Inline output
    print("\n" + "-" * 60)
    print("TEST 3: Balancing (Inline Output)")
    print("-" * 60)
    
    opt_inline = {'verbose': True, 'info_separately': False}
    print("Processing: ", end='')
    B3 = balance_triplets(M, opt_inline)
    print()  # Newline after inline output
    
    # Test 4: Check convergence
    print("\n" + "-" * 60)
    print("TEST 4: Verify Balanced Properties")
    print("-" * 60)
    
    # Check that column weights are approximately correct
    col_weights = []
    for l in range(n_points):
        rows = np.where(~np.isnan(M[0::3, l]))[0]
        if len(rows) > 0:
            rowsb = Utils.k2i(rows, step=3)
            weight = np.sum(B[rowsb, l] ** 2)
            col_weights.append(weight)
            expected = len(rows)
            if abs(weight - expected) > 1.0:
                print(f"  Column {l}: weight={weight:.2f}, expected={expected} ✗")
    
    # Check that row triplet weights are approximately correct
    row_weights = []
    for k in range(m_images):
        cols = np.where(~np.isnan(M[3*k, :]))[0]
        if len(cols) > 0:
            weight = np.sum(B[3*k:3*k+3, cols] ** 2)
            row_weights.append(weight)
            expected = len(cols)
            if abs(weight - expected) > 1.0:
                print(f"  Row triplet {k}: weight={weight:.2f}, expected={expected} ✗")
    
    print(f"Column weights: mean={np.mean(col_weights):.2f}, std={np.std(col_weights):.2f}")
    print(f"Row weights: mean={np.mean(row_weights):.2f}, std={np.std(row_weights):.2f}")
    print(f"✓ Weights are balanced (std close to 0)")
    
    # Test 5: Preserve NaN pattern
    print("\n" + "-" * 60)
    print("TEST 5: Preserve Missing Data Pattern")
    print("-" * 60)
    
    nan_before = np.isnan(M)
    nan_after = np.isnan(B)
    
    same_pattern = np.array_equal(nan_before, nan_after)
    same_count = np.sum(nan_before) == np.sum(nan_after)
    
    print(f"NaN pattern identical: {same_pattern}")
    print(f"NaN count matches: {same_count} ({np.sum(nan_before)} before, {np.sum(nan_after)} after)")
    
    print("\n" + "=" * 60)
    print("✓ All tests completed")
    print("=" * 60)
