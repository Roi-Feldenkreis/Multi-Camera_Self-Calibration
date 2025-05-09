import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from RANSAC_Algorithm import REG
from RANSAC_Algorithm import fsampson


def generate_synthetic_data(num_points=100, outlier_ratio=0.3, noise_level=1.0):
    """
    Generate synthetic 3D points and project them onto two camera views
    with added noise and outliers to simulate realistic data.

    Parameters:
    -----------
    num_points : int, optional (default=100)
        Number of 3D points to generate
    outlier_ratio : float, optional (default=0.3)
        Percentage of outliers to include
    noise_level : float, optional (default=1.0)
        Standard deviation of Gaussian noise to add to points

    Returns:
    --------
    pts3d : numpy.ndarray (3xN)
        3D points
    pts_img1 : numpy.ndarray (3xN)
        Points projected to first camera (homogeneous coordinates)
    pts_img2 : numpy.ndarray (3xN)
        Points projected to second camera (homogeneous coordinates)
    true_F : numpy.ndarray (3x3)
        True fundamental matrix
    """
    # Generate random 3D points in a cube
    pts3d = np.random.rand(3, num_points) * 10 - 5

    # Define camera matrices for two views
    # First camera at origin
    P1 = np.array([
        [1000, 0, 500, 0],
        [0, 1000, 500, 0],
        [0, 0, 1, 0]
    ])

    # Second camera translated and rotated
    # Translation along X-axis by 2 units
    t = np.array([2, 0, 0])

    # Rotation around Y-axis by 15 degrees
    angle = np.deg2rad(15)
    R = np.array([
        [np.cos(angle), 0, np.sin(angle)],
        [0, 1, 0],
        [-np.sin(angle), 0, np.cos(angle)]
    ])

    # Essential matrix from rotation and translation
    tx = np.array([
        [0, -t[2], t[1]],
        [t[2], 0, -t[0]],
        [-t[1], t[0], 0]
    ])
    E = tx @ R

    # Second camera matrix
    P2 = np.zeros((3, 4))
    P2[:, :3] = R
    P2[:, 3] = t
    P2 = P1[:, :3] @ P2  # Apply internal camera parameters

    # Project 3D points to both cameras
    pts_h = np.vstack((pts3d, np.ones(num_points)))
    pts_img1 = P1 @ pts_h
    pts_img2 = P2 @ pts_h

    # Normalize to get homogeneous coordinates
    pts_img1 = pts_img1 / pts_img1[2, :]
    pts_img2 = pts_img2 / pts_img2[2, :]

    # Add Gaussian noise to simulate measurement errors
    pts_img1[:2, :] += np.random.normal(0, noise_level, (2, num_points))
    pts_img2[:2, :] += np.random.normal(0, noise_level, (2, num_points))

    # Calculate true fundamental matrix
    # F = inv(K2).T @ E @ inv(K1)
    K = P1[:, :3]  # Assuming same K for both cameras
    true_F = np.linalg.inv(K).T @ E @ np.linalg.inv(K)

    # Normalize F
    true_F = true_F / np.linalg.norm(true_F)

    # Add outliers by replacing some point pairs with random ones
    num_outliers = int(num_points * outlier_ratio)
    outlier_indices = np.random.choice(num_points, num_outliers, replace=False)
    pts_img2[:2, outlier_indices] = np.random.rand(2, num_outliers) * 1000

    return pts3d, pts_img1, pts_img2, true_F


def draw_epipolar_lines(img1, img2, F, pts1, pts2, inliers=None, num_lines=10):
    """
    Draw epipolar lines on the images

    Parameters:
    -----------
    img1, img2 : numpy.ndarray
        Images to draw on (or None to create blank images)
    F : numpy.ndarray (3x3)
        Fundamental matrix
    pts1, pts2 : numpy.ndarray (3xN)
        Point correspondences in homogeneous coordinates
    inliers : numpy.ndarray (N,) of bool, optional
        Boolean mask indicating inliers
    num_lines : int, optional (default=10)
        Number of epipolar lines to draw
    """
    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    # Set image dimensions (or use actual images if provided)
    img_height, img_width = 1000, 1000
    if img1 is None:
        img1 = np.ones((img_height, img_width, 3))
    if img2 is None:
        img2 = np.ones((img_height, img_width, 3))

    # Display images
    axes[0].imshow(img1)
    axes[1].imshow(img2)

    # If inliers mask is provided, use it to select points
    if inliers is not None:
        pts1_inliers = pts1[:, inliers]
        pts2_inliers = pts2[:, inliers]
        if np.sum(inliers) > num_lines:
            indices = np.random.choice(np.sum(inliers), num_lines, replace=False)
            pts1_sel = pts1_inliers[:, indices]
            pts2_sel = pts2_inliers[:, indices]
        else:
            pts1_sel = pts1_inliers
            pts2_sel = pts2_inliers
    else:
        if pts1.shape[1] > num_lines:
            indices = np.random.choice(pts1.shape[1], num_lines, replace=False)
            pts1_sel = pts1[:, indices]
            pts2_sel = pts2[:, indices]
        else:
            pts1_sel = pts1
            pts2_sel = pts2

    # Plot points in both images
    axes[0].scatter(pts1_sel[0, :], pts1_sel[1, :], c='b', marker='o')
    axes[1].scatter(pts2_sel[0, :], pts2_sel[1, :], c='b', marker='o')

    # Draw epipolar lines
    for i in range(pts1_sel.shape[1]):
        # For each point in the first image
        # Calculate the corresponding epipolar line in the second image
        line = F @ pts1_sel[:, i]

        # Line equation: ax + by + c = 0
        a, b, c = line

        # Compute endpoints of the line in the image
        if abs(b) > 1e-8:
            x0, x1 = 0, img_width
            y0 = (-c - a * x0) / b
            y1 = (-c - a * x1) / b
            axes[1].plot([x0, x1], [y0, y1], 'r-')
        else:
            # Vertical line
            x0 = -c / a
            axes[1].axvline(x0, color='r')

        # For each point in the second image
        # Calculate the corresponding epipolar line in the first image
        line = F.T @ pts2_sel[:, i]

        # Line equation: ax + by + c = 0
        a, b, c = line

        # Compute endpoints of the line in the image
        if abs(b) > 1e-8:
            x0, x1 = 0, img_width
            y0 = (-c - a * x0) / b
            y1 = (-c - a * x1) / b
            axes[0].plot([x0, x1], [y0, y1], 'g-')
        else:
            # Vertical line
            x0 = -c / a
            axes[0].axvline(x0, color='g')

    # Set titles
    axes[0].set_title('Image 1 with Epipolar Lines')
    axes[1].set_title('Image 2 with Epipolar Lines')

    # Adjust display limits
    axes[0].set_xlim(0, img_width)
    axes[0].set_ylim(img_height, 0)  # Invert y-axis for image coordinates
    axes[1].set_xlim(0, img_width)
    axes[1].set_ylim(img_height, 0)  # Invert y-axis for image coordinates

    plt.tight_layout()
    plt.show()


def plot_3d_scene(pts3d, camera1, camera2):
    """
    Visualize the 3D points and camera positions

    Parameters:
    -----------
    pts3d : numpy.ndarray (3xN)
        3D points
    camera1, camera2 : numpy.ndarray (3x4)
        Camera matrices
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot 3D points
    ax.scatter(pts3d[0, :], pts3d[1, :], pts3d[2, :], c='b', marker='o')

    # Plot camera positions
    # For simplicity, we just plot the camera centers
    # Camera 1 is at the origin
    ax.scatter([0], [0], [0], c='r', marker='^', s=100, label='Camera 1')

    # Camera 2 position (simplified for this example)
    ax.scatter([2], [0], [0], c='g', marker='^', s=100, label='Camera 2')

    # Add axis labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D Scene with Cameras')
    ax.legend()

    # Set equal aspect ratio
    ax.set_box_aspect([1, 1, 1])

    plt.show()


def main():
    # Generate synthetic data
    print("Generating synthetic data...")
    pts3d, pts_img1, pts_img2, true_F = generate_synthetic_data(
        num_points=200, outlier_ratio=0.3, noise_level=1.0
    )

    # Stack points into the format expected by the RANSAC function
    points = np.vstack((pts_img1, pts_img2))

    # Estimate fundamental matrix using RANSAC
    print("Estimating fundamental matrix using RANSAC...")
    estimated_F, inliers = REG.reg(points, th=2.0, conf=0.99, ss=8)

    # Normalize the fundamental matrix
    estimated_F = estimated_F / np.linalg.norm(estimated_F)

    # Calculate errors using the estimated F
    errors = fsampson(estimated_F, points)

    print(f"True Fundamental Matrix:\n{true_F}")
    print(f"\nEstimated Fundamental Matrix:\n{estimated_F}")
    print(f"\nNumber of inliers: {np.sum(inliers)} out of {points.shape[1]} points")
    print(f"Average Sampson error for inliers: {np.mean(errors[inliers]):.4f}")

    # Plot 3D scene
    plot_3d_scene(pts3d, None, None)

    # Draw epipolar lines
    draw_epipolar_lines(None, None, estimated_F, pts_img1, pts_img2, inliers, num_lines=15)

    # Compare with epipolar lines from true F
    draw_epipolar_lines(None, None, true_F, pts_img1, pts_img2, inliers, num_lines=15)


if __name__ == "__main__":
    main()