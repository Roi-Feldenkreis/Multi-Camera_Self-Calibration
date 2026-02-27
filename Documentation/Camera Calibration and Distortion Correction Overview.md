# Camera Calibration and Distortion Correction Overview

## Introduction

This collection of files implements camera calibration and distortion correction methods, primarily focused on handling radial and tangential lens distortion. The code is based on established techniques from computer vision research,
particularly incorporating models from the University of Oulu and influenced by the CalTech camera calibration toolbox.

## Purpose

Real-world camera lenses introduce distortion to images, causing straight lines to appear curved. These distortions must be corrected for accurate computer vision applications such as:
- 3D reconstruction
- Structure from motion
- Camera pose estimation
- Accurate measurements from images

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

## File Descriptions

### 1. comp_distortion_oulu.py
This function compensates for radial and tangential distortion using an iterative approach.
It takes distorted normalized coordinates and distortion coefficients as inputs, producing undistorted coordinates as output.

Key features:
- Implements the Oulu University distortion model
- Uses an iterative method running for 20 iterations to achieve convergence
- Handles both radial distortion (k1, k2, k3) and tangential distortion (p1, p2)

### 2. isptnorm.py
This function performs isotropic point normalization,
which is a preprocessing step often used in computer vision algorithms to improve numerical stability.

Key features:
- Creates homogeneous coordinates
- Normalizes the coordinates by scaling and translating them
- Returns both normalized coordinates and the transformation matrix

### 3. readradfile.py
This function reads calibration parameters from ".rad" files used in the BlueC system.

Key features:
- Parses files containing camera calibration information
- Returns the 3×3 calibration matrix K and 4×1 vector of distortion parameters kc

### 4. undoheikk.py (imcorr.py)
This function corrects image coordinates contaminated by radial and tangential distortion,
implementing a model developed by Janne Heikkila from the University of Oulu.

Key features:
- Takes system parameters, camera intrinsic parameters, and distorted image coordinates
- Computes the correction for radial and tangential distortion
- Returns corrected image coordinates

### 5. undoradial.py
This function removes radial distortion from pixel coordinates,
transforming them to follow the linear pinhole camera model.

Key features:
- Extracts principal point and focal length from the calibration matrix
- Normalizes coordinates by subtracting the principal point and dividing by focal length
- Calls comp_distortion_oulu.py to perform the actual undistortion
- Transforms the normalized coordinates back to pixel coordinates

## Mathematical Model

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

The undistortion process (implemented in comp_distortion_oulu.py and used by undoradial.py) essentially reverses this process through iteration.

## Usage Workflow

A typical usage workflow for these functions:

1. Calibrate a camera to obtain K and kc parameters (or read them from .rad files using readradfile.py)
2. For distorted pixel coordinates in an image:
   - Use undoradial.py to transform them to undistorted pixel coordinates
   - Or use undoheikk.py (imcorr.py) for a different implementation

## Applications

This code would be useful for:
- Camera calibration processes
- Pre-processing images for computer vision algorithms
- 3D reconstruction from multiple views
- Photogrammetry applications
- Any application requiring undistorted images or coordinates

## References

The code references and appears to be influenced by:
- The University of Oulu camera calibration research (Janne Heikkila)
- The CalTech camera calibration toolbox
