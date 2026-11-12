"""
camera_calibration_visualizer.py - Main interface for camera calibration visualization tools

This module provides a convenient interface to the camera calibration visualization tools
converted from MATLAB to Python. It serves as an overview file that demonstrates how to use
the individual components together.

Components included:
- draw_cloud: Draws a 3D point cloud
- draw_object: Draws calibration objects (cube, octagon, point cloud)
- draw_scene: Visualizes a scene with calibration points, cameras, and viewing axes
- show_img: Displays scaled images in grayscale

Original MATLAB Info:
The functions in this package were originally developed for camera calibration
visualization purposes, authored primarily by Svoboda and Pajdla.

Example usage is provided at the end of this file.
"""

import numpy as np
import matplotlib.pyplot as plt

from OutputFunctions.Draw_Cloud import draw_cloud
from OutputFunctions.Draw_Object import draw_object
from OutputFunctions.Draw_Scene import draw_scene
from OutputFunctions.Show_Img import show_img


def visualize_calibration_example():
    """
    Demonstrate how to use the camera calibration visualization tools.

    This function creates sample data and calls the visualization functions
    to show their functionality.
    """
    print("Camera Calibration Visualization Toolkit Example")
    print("-----------------------------------------------")

    # Generate example data
    POINTS = 20  # Number of points in calibration object
    CAMS = 5  # Number of cameras

    # Create random calibration points (4xPOINTS matrix)
    X = np.random.rand(4, POINTS)
    # Make the last row all 1s (homogeneous coordinates)
    X[3, :] = 1.0

    # Create random camera centers (3xCAMS matrix)
    C = np.random.rand(3, CAMS) * 5

    # Create rotation matrices (3*CAMSx3 matrix)
    R = np.zeros((3 * CAMS, 3))
    for i in range(CAMS):
        # Create a random rotation matrix
        temp = np.random.rand(3, 3)
        # Make it orthogonal (a proper rotation matrix)
        q, r = np.linalg.qr(temp)
        # Ensure it's a proper rotation matrix (det=1)
        if np.linalg.det(q) < 0:
            q[:, 0] = -q[:, 0]  # Flip one column if needed
        # Store it in the larger matrix
        R[3 * i:3 * (i + 1), :] = q

    # Example 1: Drawing a point cloud
    print("\n1. Drawing a point cloud...")
    fig_cloud = plt.figure(figsize=(8, 6))
    fig_cloud = draw_cloud(X, fig_cloud, 'g')
    plt.title("Example Point Cloud")

    # Example 2: Drawing calibration objects
    print("\n2. Drawing calibration objects...")
    # Create vertices for a cube
    cube_vertices = np.array([
        [0, 1, 1, 0, 0, 1, 1, 0],  # x coordinates
        [0, 0, 1, 1, 0, 0, 1, 1],  # y coordinates
        [0, 0, 0, 0, 1, 1, 1, 1]  # z coordinates
    ])

    # Create vertices for an octagon (in 3D, but planar)
    t = np.linspace(0, 2 * np.pi, 9)[:8]  # 8 points around a circle
    octagon_vertices = np.array([
        np.cos(t),  # x coordinates
        np.sin(t),  # y coordinates
        np.zeros_like(t)  # z coordinates (all 0 for planar shape)
    ])

    # Draw the objects
    fig_objects = plt.figure(figsize=(15, 5))

    # Create the cube subplot
    plt.subplot(131, projection='3d')
    draw_object(cube_vertices, 'cube', fig_objects, 'r')
    plt.title("Cube Calibration Object")

    # Create the octagon subplot
    plt.subplot(132, projection='3d')
    draw_object(octagon_vertices, 'octagon', fig_objects, 'g')
    plt.title("Octagon Calibration Object")

    # Create the point cloud subplot
    plt.subplot(133, projection='3d')
    draw_object(X[:3, :], 'cloud', fig_objects, 'b')
    plt.title("Point Cloud Calibration Object")

    # Example 3: Visualizing a complete calibration scene
    print("\n3. Visualizing a complete calibration scene...")
    fig_scene = plt.figure(figsize=(10, 8))
    fig_scene = draw_scene(X, C, R, fig_scene, 'cloud', 'Complete Calibration Scene')

    # Example 4: Displaying a grayscale image
    print("\n4. Displaying a grayscale image...")
    # Create a sample image (gradient with some pattern)
    x = np.linspace(-3, 3, 200)
    y = np.linspace(-3, 3, 200)
    X_grid, Y_grid = np.meshgrid(x, y)
    Z = np.sin(X_grid ** 2 + Y_grid ** 2) / (X_grid ** 2 + Y_grid ** 2 + 0.1)

    fig_img = plt.figure(figsize=(8, 6))
    fig_img = show_img(Z, fig_img)
    plt.title("Sample Grayscale Image")
    plt.colorbar()

    print("\nAll visualizations created. Close the windows to exit.")
    plt.show()


if __name__ == "__main__":
    # Run the example visualization
    visualize_calibration_example()