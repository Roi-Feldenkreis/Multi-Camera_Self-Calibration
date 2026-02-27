# Fundamental Matrix Estimation

This package provides Python implementations of algorithms for estimating the fundamental matrix in computer vision.

## Overview

The fundamental matrix is a 3×3 matrix that encapsulates the epipolar geometry between two camera views. For any point in one image, its corresponding point in the other image must lie on the epipolar line defined by the fundamental matrix.

This package implements a robust pipeline for fundamental matrix estimation using RANSAC (Random Sample Consensus) and DLT.

## Files Description

1. **fsampson.py**: Calculates the Sampson distance, a first-order approximation of the geometric error for point correspondences.

2. **nsamples.py**: Computes the optimal number of RANSAC iterations based on the current inlier ratio.

3. **pointnormiso.py**: Performs isotropic normalization of points to improve numerical stability.

4. **u2fdlt.py**: Implements the Direct Linear Transform (DLT) algorithm for estimating the fundamental matrix.

5. **reg.py**: Implements RANSAC for robust fundamental matrix estimation in the presence of outliers.

6. **fundamental_matrix_example.py**: A complete example demonstrating how to use these functions.

## Requirements

- NumPy
- Matplotlib (for visualization only in the example)

## Usage

The main function for estimating the fundamental matrix is `reg()` which takes a set of corresponding points and returns the estimated fundamental matrix along with a mask indicating inliers.

```python
from REG import reg

# u is a 6xN array where the first 3 rows are points in the first image
# and the last 3 rows are corresponding points in the second image
# (all in homogeneous coordinates)
F, inliers = reg(u, th=2.0)

# F is the estimated 3x3 fundamental matrix
# inliers is a boolean array indicating which points are inliers
```

See `fundamental_matrix_example.py` for a complete working example including generating synthetic data and visualization.

## Example Output

Running the example code will:
1. Generate synthetic 3D points and project them onto two camera views
2. Add noise and outliers to simulate real-world conditions
3. Estimate the fundamental matrix using RANSAC
4. Visualize the 3D scene and epipolar lines

## Theoretical Background

### Epipolar Geometry

Epipolar geometry describes the geometric relationship between two camera views of the same scene. The fundamental matrix F satisfies the epipolar constraint:

```
	x2^T · F · x1 = 0
```

where x1 and x2 are corresponding points in the two images (in homogeneous coordinates).

### Robust Estimation

RANSAC is used for robust estimation in the presence of outliers:
	1. Randomly select a minimal subset of correspondences the default is 8.
	2. Estimate a model (fundamental matrix) from this subset
	3. Count the number of inliers (points that satisfy the model within a threshold)
	4. Repeat and keep the model with the most inliers
	5. Refine the model using all inliers

## Credits

Original MATLAB implementation by [Svoboda]. Python conversion and documentation by [Roi Feldenkreis].