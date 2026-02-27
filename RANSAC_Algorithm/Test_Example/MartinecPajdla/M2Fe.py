"""
M2Fe - Estimate epipolar geometry in sequence or using central image
Converted from MATLAB to Python
"""

import numpy as np
from typing import Tuple, Dict, List
from MartinecPajdla.Utils import Utils
from MartinecPajdla.u2FI import u2FI


def M2Fe(M: np.ndarray, central: int = 0) -> Tuple[Dict, Dict, np.ndarray, np.ndarray]:
    """
    Estimate epipolar geometry of multiple views in sequence or using central image.
    
    Args:
        M: (3*m, N) array where m is number of images, each image takes 3 rows
           Each image's points are in homogeneous coordinates [x, y, 1]
        central: If 0, use sequence mode (consecutive pairs)
                 If > 0, use central image mode (all vs image 'central')
    
    Returns:
        F: Dictionary of fundamental matrices {(i,j): F_ij}
        ep: Dictionary of epipoles {(i,j): epipole_ij}
        rows: Array of valid image indices
        nonrows: Array of failed image indices
    """
    
    # Get number of images
    m = M.shape[0] // 3
    nonrows = []
    F = {}
    ep = {}
    
    # Determine which image pairs to process
    if central > 0:
        # Central image mode: compare all images to central image
        # Skip the central image itself
        rows = np.concatenate([np.arange(central), np.arange(central + 1, m)])
    else:
        # Sequence mode: compare consecutive images
        rows = np.arange(1, m)
    
    rows_list = list(rows)
    
    # Estimate fundamental matrices and epipoles
    for k in rows:
        if central > 0:
            j = central
        else:
            j = k - 1
        
        # Extract point correspondences for images k and j
        # MATLAB: M(3*k-2:3*k,:) → Python: M[3*k:3*k+3,:]
        u_pair = np.vstack([M[3*k:3*k+3, :], M[3*j:3*j+3, :]])
        
        # Estimate fundamental matrix
        G = u2FI(u_pair, normalization='norm')
        
        if isinstance(G, int) and G == 0:
            # Failed to estimate F for this pair
            rows_list.remove(k)
            nonrows.append(k)
        else:
            # Compute epipole using SVD
            # epipole is the null space of G' (right null space of G)
            u, s, vt = np.linalg.svd(G)
            epip = u[:, 2]  # Last column of U (smallest singular value)
            
            # Store fundamental matrix and epipole
            F[(k, j)] = G
            ep[(k, j)] = epip
    
    rows = np.array(rows_list)
    
    # Post-processing
    if len(rows) == 0:
        # All images failed - return empty
        nonrows = np.arange(m)
    elif central > 0:
        # Add central image back to valid rows
        rows = np.union1d(rows, [central])
    else:
        # Add first image to rows in sequence mode
        rows = np.concatenate([[0], rows])
    
    # If sequence mode and there are failures, find longest continuous subsequence
    if len(nonrows) > 0 and central == 0 and len(rows) > 0:
        # Create indicator array
        I = np.zeros((m, 1), dtype=bool)
        I[rows.astype(int)] = True
        
        # Find longest continuous subsequence
        b, length = Utils.subseq_longest(I)
        b = b[0]  # Get starting position
        length = length[0]  # Get length
        
        rows = np.arange(b, b + length)
        nonrows = np.setdiff1d(np.arange(m), rows)
    
    nonrows = np.array(nonrows)
    
    return F, ep, rows, nonrows


if __name__ == "__main__":
    # Test M2Fe function
    print("=" * 60)
    print("M2Fe - Epipolar Geometry Estimation Test")
    print("=" * 60)
    
    np.random.seed(42)
    
    # Generate synthetic multi-view data
    n_images = 5
    n_points = 20
    
    # Simulate 3D points
    X = np.random.randn(3, n_points) * 100
    X[2, :] += 500
    
    # Camera parameters
    K = np.array([[800, 0, 320], [0, 800, 240], [0, 0, 1]])
    
    # Generate multiple camera poses
    M_all = []
    for i in range(n_images):
        # Small rotation and translation for each camera
        theta = i * 0.05
        R = np.array([[np.cos(theta), -np.sin(theta), 0],
                      [np.sin(theta), np.cos(theta), 0],
                      [0, 0, 1]])
        t = np.array([[i * 30], [i * 5], [0]])
        
        # Projection matrix
        P = K @ np.hstack([R, t])
        
        # Project 3D points
        X_homo = np.vstack([X, np.ones((1, n_points))])
        u = P @ X_homo
        u = u / u[2, :]
        
        M_all.append(u)
    
    # Stack all images into M matrix
    M = np.vstack(M_all)
    
    print(f"\nGenerated data: {n_images} images, {n_points} points")
    print(f"M shape: {M.shape} (each image = 3 rows)")
    
    # Test 1: Sequence mode
    print("\n" + "-" * 60)
    print("TEST 1: Sequence Mode (consecutive pairs)")
    print("-" * 60)
    
    F_seq, ep_seq, rows_seq, nonrows_seq = M2Fe(M, central=0)
    
    print(f"Valid images: {rows_seq}")
    print(f"Failed images: {nonrows_seq}")
    print(f"Number of F matrices: {len(F_seq)}")
    
    if len(F_seq) > 0:
        # Show first fundamental matrix
        first_key = list(F_seq.keys())[0]
        print(f"\nFirst F matrix (images {first_key}):")
        print(F_seq[first_key])
        print(f"Rank: {np.linalg.matrix_rank(F_seq[first_key], tol=1e-6)}")
        print(f"Epipole shape: {ep_seq[first_key].shape}")
    
    # Test 2: Central image mode
    print("\n" + "-" * 60)
    print("TEST 2: Central Image Mode (all vs image 2)")
    print("-" * 60)
    
    F_cent, ep_cent, rows_cent, nonrows_cent = M2Fe(M, central=2)
    
    print(f"Valid images: {rows_cent}")
    print(f"Failed images: {nonrows_cent}")
    print(f"Number of F matrices: {len(F_cent)}")
    
    if len(F_cent) > 0:
        # Show first fundamental matrix
        first_key = list(F_cent.keys())[0]
        print(f"\nFirst F matrix (images {first_key}):")
        print(F_cent[first_key])
        print(f"Rank: {np.linalg.matrix_rank(F_cent[first_key], tol=1e-6)}")
    
    # Test 3: With insufficient data
    print("\n" + "-" * 60)
    print("TEST 3: With Insufficient Data")
    print("-" * 60)
    
    # Create data with only 5 points (insufficient for F estimation)
    M_small = M[:, :5]
    
    F_small, ep_small, rows_small, nonrows_small = M2Fe(M_small, central=0)
    
    print(f"Valid images: {rows_small}")
    print(f"Failed images: {nonrows_small}")
    print(f"Number of F matrices: {len(F_small)}")
    
    print("\n" + "=" * 60)
    print("✓ All tests completed")
    print("=" * 60)
