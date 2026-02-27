"""
depth_estimation - Determine scale factors (projective depths) of PRMM
Converted from MATLAB to Python
"""

import numpy as np
from typing import Tuple, Dict
from MartinecPajdla.Utils import Utils


def depth_estimation(M: np.ndarray, 
                    F: Dict, 
                    ep: Dict, 
                    rows: np.ndarray, 
                    central: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Determine scale factors (projective depths) for multi-view reconstruction.
    
    Args:
        M: (3*m, n) array of measurements (m images, n points)
        F: Dictionary of fundamental matrices from M2Fe
        ep: Dictionary of epipoles from M2Fe
        rows: Valid image indices from M2Fe
        central: If 0, sequence mode; if >0, central image number
    
    Returns:
        lambda: (m, n) array of scale factors (projective depths)
        Ilamb: (m, n) boolean array indicating valid depths
    """
    
    # Get dimensions
    m = M.shape[0] // 3  # number of images
    n = M.shape[1]       # number of points
    
    # Initialize outputs
    lambda_vals = np.ones((m, n))
    Ilamb = np.zeros((m, n), dtype=bool)
    
    # Determine reference image and point sets
    if central > 0:
        # Central mode: use central image as reference
        j = central
        ps = np.arange(n)  # all points
        Ilamb[j, :] = ~np.isnan(M[3*j, :])
    else:
        # Sequence mode: use first image as reference
        j = 0
        # Find longest subsequence for each point
        valid_mask = ~np.isnan(M[0::3, :])  # Check x-coordinates
        b, _ = Utils.subseq_longest(valid_mask)
        
        for p in range(n):
            img_idx = b[p]
            Ilamb[img_idx, p] = ~np.isnan(M[3*img_idx, p])
    
    # Process each image except reference
    for i in np.setdiff1d(np.arange(m), [j]):
        # Update reference for sequence mode
        if central == 0:
            j = i - 1
            # Points where reference image has data
            ps = np.where(b <= j)[0]

        # M2Fe stores F and ep keyed by ACTUAL camera indices (rows[i], rows[j]),
        # not local loop indices. Must translate local -> actual before lookup.
        actual_i = rows[i]
        actual_j = rows[j]

        # Check if this image pair has F matrix
        if (actual_i, actual_j) not in F:
            continue
        
        # Get fundamental matrix and epipole
        G = F[(actual_i, actual_j)]
        epip = ep[(actual_i, actual_j)].reshape(3, 1)
        
        # Process each point
        for p in ps:
            # Point is valid if both images have it
            Ilamb[i, p] = Ilamb[j, p] and ~np.isnan(M[3*i, p])
            
            if Ilamb[i, p]:
                # Extract points from both images
                u_i = M[3*i:3*i+3, p].reshape(3, 1)
                u_j = M[3*j:3*j+3, p].reshape(3, 1)
                
                # Compute depth factor
                # u = epipole × point_i (cross product gives epipolar line)
                u = np.cross(epip.flatten(), u_i.flatten()).reshape(3, 1)
                
                # v = F * point_j (transforms to epipolar line in image i)
                v = G @ u_j
                
                # Compute scale factor
                u_norm_sq = np.linalg.norm(u)**2
                if u_norm_sq > 1e-10:
                    lambda_vals[i, p] = (u.T @ v / u_norm_sq * lambda_vals[j, p]).item()
                else:
                    lambda_vals[i, p] = 1.0
            else:
                # No valid correspondence - set to 1 for later recovery
                lambda_vals[i, p] = 1.0
    
    return lambda_vals, Ilamb


if __name__ == "__main__":
    # Test depth_estimation function
    print("=" * 60)
    print("depth_estimation - Scale Factor Estimation Test")
    print("=" * 60)
    
    from M2Fe import M2Fe
    
    np.random.seed(42)
    
    # Generate synthetic multi-view data with known depths
    n_images = 4
    n_points = 15
    
    # True 3D points
    X_true = np.random.randn(3, n_points) * 100
    X_true[2, :] += 500  # Move away from camera
    
    # Camera parameters
    K = np.array([[800, 0, 320], [0, 800, 240], [0, 0, 1]])
    
    # Generate multiple views
    M_all = []
    true_depths = []
    
    for i in range(n_images):
        # Camera pose
        theta = i * 0.05
        R = np.array([[np.cos(theta), -np.sin(theta), 0],
                      [np.sin(theta), np.cos(theta), 0],
                      [0, 0, 1]])
        t = np.array([[i * 30], [i * 5], [0]])
        
        # Projection matrix
        P = K @ np.hstack([R, t])
        
        # Project 3D points
        X_homo = np.vstack([X_true, np.ones((1, n_points))])
        u = P @ X_homo
        
        # Store depths before normalization
        depths = u[2, :]
        true_depths.append(depths)
        
        # Normalize to homogeneous coordinates
        u = u / u[2, :]
        M_all.append(u)
    
    # Stack into M matrix
    M = np.vstack(M_all)
    
    print(f"\nGenerated data: {n_images} images, {n_points} points")
    print(f"M shape: {M.shape}")
    
    # Test 1: Sequence mode
    print("\n" + "-" * 60)
    print("TEST 1: Sequence Mode")
    print("-" * 60)
    
    F_seq, ep_seq, rows_seq, nonrows_seq = M2Fe(M, central=0)
    
    if len(F_seq) > 0:
        lambda_seq, Ilamb_seq = depth_estimation(M, F_seq, ep_seq, rows_seq, central=0)
        
        print(f"Lambda shape: {lambda_seq.shape}")
        print(f"Valid depths: {np.sum(Ilamb_seq)}/{Ilamb_seq.size}")
        
        print(f"\nSample lambda values (first 5 points):")
        for i in range(min(4, n_images)):
            print(f"Image {i}: {lambda_seq[i, :5]}")
        
        # Check that reference image has lambda=1
        print(f"\nReference image (0) lambda values: {lambda_seq[0, :5]}")
        print(f"All ones? {np.allclose(lambda_seq[0, :], 1.0)}")
    else:
        print("No fundamental matrices estimated")
    
    # Test 2: Central mode
    print("\n" + "-" * 60)
    print("TEST 2: Central Image Mode (central=1)")
    print("-" * 60)
    
    F_cent, ep_cent, rows_cent, nonrows_cent = M2Fe(M, central=1)
    
    if len(F_cent) > 0:
        lambda_cent, Ilamb_cent = depth_estimation(M, F_cent, ep_cent, rows_cent, central=1)
        
        print(f"Lambda shape: {lambda_cent.shape}")
        print(f"Valid depths: {np.sum(Ilamb_cent)}/{Ilamb_cent.size}")
        
        print(f"\nSample lambda values (first 5 points):")
        for i in range(min(4, n_images)):
            print(f"Image {i}: {lambda_cent[i, :5]}")
        
        # Check that central image has lambda=1
        print(f"\nCentral image (1) lambda values: {lambda_cent[1, :5]}")
        print(f"All ones? {np.allclose(lambda_cent[1, :], 1.0)}")
    else:
        print("No fundamental matrices estimated")
    
    # Test 3: Check validity indicators
    print("\n" + "-" * 60)
    print("TEST 3: Validity Indicators")
    print("-" * 60)
    
    if len(F_seq) > 0:
        print(f"Ilamb shape: {Ilamb_seq.shape}")
        print(f"Valid entries per image:")
        for i in range(n_images):
            n_valid = np.sum(Ilamb_seq[i, :])
            print(f"  Image {i}: {n_valid}/{n_points} valid")
    
    # Test 4: With missing data
    print("\n" + "-" * 60)
    print("TEST 4: With Missing Data (NaN values)")
    print("-" * 60)
    
    # Add some NaN values
    M_missing = M.copy()
    M_missing[3:6, 5] = np.nan  # Remove point 5 from image 1
    M_missing[6:9, 7] = np.nan  # Remove point 7 from image 2
    
    F_miss, ep_miss, rows_miss, _ = M2Fe(M_missing, central=0)
    
    if len(F_miss) > 0:
        lambda_miss, Ilamb_miss = depth_estimation(M_missing, F_miss, ep_miss, rows_miss, central=0)
        
        print(f"Valid depths: {np.sum(Ilamb_miss)}/{Ilamb_miss.size}")
        print(f"\nPoint 5 validity (should be False for image 1):")
        print(f"  Image 0: {Ilamb_miss[0, 5]}")
        print(f"  Image 1: {Ilamb_miss[1, 5]}")
        print(f"  Image 2: {Ilamb_miss[2, 5]}")
    
    print("\n" + "=" * 60)
    print("✓ All tests completed")
    print("=" * 60)
