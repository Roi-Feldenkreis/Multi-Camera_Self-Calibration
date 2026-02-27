# Multi-View 3D Reconstruction System: Camera Calibration and Processing

## Introduction

This collection of code implements a comprehensive multi-view 3D reconstruction system using computer vision techniques. The system handles camera calibration, distortion correction, and 3D point reconstruction from multiple camera views. The code is based on established techniques from computer vision research, particularly incorporating models from the University of Oulu and influenced by the CalTech camera calibration toolbox.

## System Overview

This codebase provides tools for a complete multi-view 3D reconstruction pipeline:

1. **Camera Calibration**: Functions for handling camera parameters and distortion
2. **Distortion Correction**: Methods to remove lens distortion from image coordinates
3. **Working Volume Analysis**: Determination of 3D space visible from all cameras
4. **3D Point Reconstruction**: Robust estimation of 3D points from multiple views
5. **Camera Matrix Decomposition**: Extraction of intrinsic and extrinsic parameters

## Key Concepts

### Camera Distortion Models

The code implements distortion models that include:

1. **Radial Distortion**: The most common form of lens distortion that increases with distance from the optical center
   - Modeled using polynomial terms (k1, k2, k3)
   
2. **Tangential Distortion**: Caused by imperfect lens alignment
   - Modeled using parameters p1 and p2

### Camera Calibration Parameters

The code works with several key camera parameters:
- **Intrinsic Matrix (K)**: A 3×3 matrix containing focal length and principal point information
- **Distortion Coefficients (kc)**: A vector containing radial and tangential distortion parameters
- **Projection Matrix (P)**: A 3×4 matrix that maps 3D world points to 2D image points
- **Rotation Matrix (R)**: A 3×3 matrix describing camera orientation
- **Translation Vector (t)**: A 3×1 vector describing camera position
- **Camera Center (C)**: The 3D location of the camera in world coordinates

### Multi-View Triangulation

A core concept in this system is triangulation - the process of determining 3D coordinates of points seen in multiple camera views:
- Uses linear methods based on the Direct Linear Transformation (DLT) algorithm
- Leverages Singular Value Decomposition (SVD) to solve the system of equations
- Minimizes reprojection error to find optimal solutions

## File Descriptions

### Camera Calibration and Distortion Correction

#### 1. comp_distortion_oulu.m
This function compensates for radial and tangential distortion using an iterative approach. It takes distorted normalized coordinates and distortion coefficients as inputs, producing undistorted coordinates as output.

Key features:
- Implements the Oulu University distortion model
- Uses an iterative method running for 20 iterations to achieve convergence
- Handles both radial distortion (k1, k2, k3) and tangential distortion (p1, p2)

#### 2. isptnorm.m
This function performs isotropic point normalization, which is a preprocessing step often used in computer vision algorithms to improve numerical stability.

Key features:
- Creates homogeneous coordinates
- Normalizes the coordinates by scaling and translating them
- Returns both normalized coordinates and the transformation matrix

#### 3. readradfile.m
This function reads calibration parameters from ".rad" files used in the BlueC system.

Key features:
- Parses files containing camera calibration information
- Returns the 3×3 calibration matrix K and 4×1 vector of distortion parameters kc

#### 4. undoheikk.m (imcorr.m)
This function corrects image coordinates contaminated by radial and tangential distortion, implementing a model developed by Janne Heikkila from the University of Oulu.

Key features:
- Takes system parameters, camera intrinsic parameters, and distorted image coordinates
- Computes the correction for radial and tangential distortion
- Returns corrected image coordinates

#### 5. undoradial.m
This function removes radial distortion from pixel coordinates, transforming them to follow the linear pinhole camera model.

Key features:
- Extracts principal point and focal length from the calibration matrix
- Normalizes coordinates by subtracting the principal point and dividing by focal length
- Calls comp_distortion_oulu to perform the actual undistortion
- Transforms the normalized coordinates back to pixel coordinates

### 3D Reconstruction Components

#### 6. uP2X Function
The uP2X function implements linear triangulation for 3D point reconstruction from multiple views.

Key features:
- Based on the Direct Linear Transformation (DLT) algorithm from Hartley and Zisserman's "Multiple View Geometry"
- Takes as input:
  - Umat: Matrix of image coordinates of 2D points across N views
  - Ps: List of camera projection matrices
- For each point:
  - Constructs a linear system where each camera contributes two rows
  - Each row represents the constraint: x_i * (P_3^T X) - (P_1^T X) = 0 and y_i * (P_3^T X) - (P_2^T X) = 0
  - Solves using SVD - the solution is the singular vector corresponding to the smallest singular value
  - Normalizes the result by dividing by the last component
- Produces accurate 3D point coordinates from multiple calibrated views

#### 7. estimateX Function
This function performs robust estimation of 3D points from multiple camera views.

Key features:
- Finds the best subset of cameras for 3D reconstruction by evaluating reprojection error
- Process:
  1. Generates all possible camera combinations based on configured sample size
  2. For each combination, identifies points visible in all cameras of that combination
  3. Reconstructs 3D points using linear triangulation (uP2X)
  4. Projects these points back to 2D and calculates reprojection errors
  5. Selects the camera combination with the lowest reprojection error
- Implements a RANSAC-like approach to find optimal camera subset
- Returns final 3D reconstruction from the best camera combination

#### 8. P2KRtC Function
This function decomposes a camera projection matrix into its intrinsic and extrinsic components.

Key features:
- Decomposes a 3×4 camera projection matrix P into:
  - K: The 3×3 calibration matrix (intrinsic parameters)
  - R: The 3×3 rotation matrix (camera orientation)
  - t: The 3×1 translation vector
  - C: The 3×1 camera center position in world coordinates
- Uses RQ decomposition (implemented as the inverse of QR decomposition)
- Ensures the diagonal elements of K are positive (canonical form)

#### 9. workvolume Function
This function computes the working volume visible from multiple cameras.

Key features:
- Determines which 3D points in a specified volume are visible from all cameras
- Process:
  1. Creates a 3D grid of points within specified room dimensions
  2. Projects these points into each camera using projection matrices
  3. Checks which points are within the image boundaries of all cameras
  4. Returns the 3D points and indices of points visible in all cameras
- Useful for determining effective capture volume and camera placement optimization

## Mathematical Models

### Distortion Model

The distortion model used can be summarized as:

1. **Normalized coordinates**: 
   - x_normalized = (x_pixel - cx) / fx
   - y_normalized = (y_pixel - cy) / fy
   
   Where (cx, cy) is the principal point and (fx, fy) are focal lengths

2. **Radial distortion**:
   - r² = x² + y²
   - k_radial = 1 + k1·r² + k2·r⁴ + k3·r⁶

3. **Tangential distortion**:
   - dx = 2·p1·x·y + p2·(r² + 2·x²)
   - dy = p1·(r² + 2·y²) + 2·p2·x·y

4. **Distorted normalized coordinates**:
   - x_distorted = x_normalized·k_radial + dx
   - y_distorted = y_normalized·k_radial + dy

5. **Distorted pixel coordinates**:
   - x_pixel = fx·x_distorted + cx
   - y_pixel = fy·y_distorted + cy

The undistortion process essentially reverses this process through iteration.

### Linear Triangulation

The triangulation model in uP2X can be expressed as:

1. For a 3D point X and its 2D projection x in a given camera with projection matrix P:
   - x = PX (in homogeneous coordinates)

2. This gives us the constraints:
   - x_i * (P_3^T X) - (P_1^T X) = 0
   - y_i * (P_3^T X) - (P_2^T X) = 0
   
   Where P_j^T is the j-th row of P.

3. These equations form the rows of matrix A, and we solve AX = 0 for the homogeneous vector X using SVD

## Usage Workflow

A typical workflow using this code would involve:

1. **Camera Calibration**:
   - Obtain camera parameters (or read from .rad files using readradfile.m)
   - Decompose projection matrices using P2KRtC to understand camera setup

2. **Distortion Correction**:
   - Use undoradial.m or undoheikk.m to correct distorted image points

3. **Working Volume Analysis**:
   - Use workvolume to determine the effective capture volume

4. **3D Reconstruction**:
   - Apply uP2X for direct triangulation from multiple views
   - Use estimateX for robust reconstruction selecting optimal camera combinations

## Applications

This code would be useful for:
- Structure from Motion (SfM) systems
- 3D reconstruction from multiple calibrated cameras
- Motion capture systems
- Visual SLAM (Simultaneous Localization and Mapping)
- Camera calibration verification
- Photogrammetry applications
- Any application requiring undistorted images or accurate 3D reconstruction

## References

The code references and appears to be influenced by:
- The University of Oulu camera calibration research (Janne Heikkila)
- The CalTech camera calibration toolbox
- Hartley and Zisserman's "Multiple View Geometry in Computer Vision"
- BlueC system (referenced in readradfile.m)
