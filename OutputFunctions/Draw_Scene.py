import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Import functions from the other modules
from OutputFunctions.Draw_Cloud import draw_cloud
from OutputFunctions.Draw_Object import draw_object


def draw_scene(X, C, R, fig=None, ctype_hint='cloud', scene_title='', cams_id=None):
    """
    Plot calibration points, cameras, and their viewing axes in 3D.

    This function visualizes a camera calibration scene by plotting:
    - Calibration object points (in red)
    - Camera positions (in blue)
    - Camera viewing axes (in black)
    - Camera ID labels

    Parameters:
    -----------
    X : numpy.ndarray
        3×POINTS or 4×POINTS matrix containing object points.
        - If 3×POINTS: Each column represents one 3D point [x, y, z]^T
        - If 4×POINTS: Each column represents one point in homogeneous coordinates [x, y, z, w]^T
        The function automatically detects the format and handles both cases.
    C : numpy.ndarray
        3×CAMS matrix containing camera centers in world coordinates.
        Each column represents one camera center [x, y, z]^T
    R : numpy.ndarray
        CAMS×3×3 array containing camera rotation matrices.
        Each R[i] represents the 3×3 rotation matrix for camera i
    fig : matplotlib.figure.Figure, optional
        Existing figure to draw on. If None, creates a new figure
    ctype_hint : str, optional
        Type hint for calibration object visualization ('cloud', 'grid', etc.)
        Default: 'cloud'
    scene_title : str, optional
        Title for the 3D plot. Default: ''
    cams_id : numpy.ndarray, optional
        1×CAMS array with camera identifiers. If None, uses 1:CAMS

    Returns:
    --------
    fig : matplotlib.figure.Figure
        Handle to the matplotlib figure containing the 3D plot

    Notes:
    ------
    - The viewing axis for each camera is computed as the negative of the 3rd row
      of its rotation matrix (which points towards the camera's viewing direction)
    - Axis length is scaled to 60% of the distance from camera to object centroid
    - Equal aspect ratio is enforced for proper 3D visualization
    - Automatically handles both 3D and homogeneous coordinate inputs
    """
    # Validate and process input dimensions
    if X.shape[0] == 3:
        # 3D coordinates - convert to homogeneous by adding row of ones
        X_homo = np.vstack([X, np.ones((1, X.shape[1]))])
        print(f"Input detected as 3D coordinates (3×{X.shape[1]}), converted to homogeneous coordinates")
    elif X.shape[0] == 4:
        # Already homogeneous coordinates
        X_homo = X.copy()
        print(f"Input detected as homogeneous coordinates (4×{X.shape[1]})")
    else:
        raise ValueError(f"X must be either 3×POINTS or 4×POINTS matrix, got {X.shape[0]}×{X.shape[1]}")

    if C.shape[0] != 3:
        raise ValueError("C must be a 3×CAMS matrix")

    # CHANGE 1: Updated validation for new R format
    if len(R.shape) != 3 or R.shape[1] != 3 or R.shape[2] != 3:
        raise ValueError(f"R must be a CAMS×3×3 array, got shape {R.shape}")

    # Extract dimensions
    POINTS = X_homo.shape[1]
    CAMS = C.shape[1]

    # CHANGE 2: Updated consistency check for new R format
    if R.shape[0] != CAMS:
        raise ValueError(
            f"R array size ({R.shape[0]}×{R.shape[1]}×{R.shape[2]}) inconsistent with number of cameras ({CAMS})")

    # Set default camera IDs if not provided
    if cams_id is None:
        cams_id = np.arange(1, CAMS + 1)
    elif len(cams_id) != CAMS:
        raise ValueError(f"cams_id length ({len(cams_id)}) must match number of cameras ({CAMS})")

    # Create or prepare the figure
    if fig is None:
        fig = plt.figure(figsize=(12, 9))
        ax = fig.add_subplot(111, projection='3d')
    else:
        # Clear existing content and create new 3D subplot
        fig.clf()
        ax = fig.add_subplot(111, projection='3d')

    # Set plot properties
    ax.set_title(scene_title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # Plot camera positions (blue points)
    fig = draw_cloud(C, fig, 'b')

    # Plot calibration object (red points/structure)
    fig = draw_object(X_homo, ctype_hint, fig, 'r')

    # Calculate centroid of all object points (excluding homogeneous coordinate)
    centroid = np.mean(X_homo[:3, :], axis=1)

    # Plot viewing axes and labels for each camera
    for i in range(CAMS):
        # CHANGE 3: Extract the i-th camera's rotation matrix from 3D array
        # Now we simply index the first dimension to get the complete 3×3 matrix
        R_i = R[i, :, :]  # This gives us the 3×3 rotation matrix for camera i

        # The camera's viewing direction is the negative of the 3rd row of R
        # This is because the 3rd row represents the camera's Z-axis direction,
        # and cameras typically look in the negative Z direction
        axis_dir = -R_i[2, :]

        # Normalize the direction vector (should already be normalized, but ensure it)
        axis_dir = axis_dir / np.linalg.norm(axis_dir)

        # Calculate axis length as 60% of distance from camera to object centroid
        # This provides a visually appropriate scale for the viewing axes
        axis_len = 0.6 * np.linalg.norm(C[:, i] - centroid)

        # Calculate the endpoint of the viewing axis
        endpoint = C[:, i] + axis_len * axis_dir

        # Draw the viewing axis as a black line
        ax.plot([C[0, i], endpoint[0]],
                [C[1, i], endpoint[1]],
                [C[2, i], endpoint[2]],
                'k-', linewidth=2, alpha=0.8)

        # Add camera ID label with better positioning to avoid overlapping
        # Position label in direction opposite to viewing axis for better visibility
        label_offset = 0.15 * axis_len
        label_pos = C[:, i] - label_offset * axis_dir
        ax.text(label_pos[0], label_pos[1], label_pos[2],
                f"Cam {cams_id[i]}",
                fontsize=9, fontweight='bold', color='darkblue',
                ha='center', va='center')

    # Set equal aspect ratio for proper 3D visualization
    set_axes_equal(ax)

    # Set equal aspect ratio for proper 3D visualization
    set_axes_equal(ax)

    # Add axis labels
    ax.set_xlabel('X', fontsize=12)
    ax.set_ylabel('Y', fontsize=12)
    ax.set_zlabel('Z', fontsize=12)

    # Add a legend in the top-right corner
    ax.text2D(0.72, 0.98, 'Blue: Cameras\nRed: Calibration Object\nBlack: Viewing Axes',
              transform=ax.transAxes, fontsize=10, verticalalignment='top',
              bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray'))

    return fig


def set_axes_equal(ax):
    """
    Set equal aspect ratio for 3D matplotlib axes with improved bounds calculation.

    This ensures that spheres appear as spheres, cubes as cubes, etc.
    Without this, the 3D plot may appear distorted due to different
    scaling on different axes.

    Parameters:
    -----------
    ax : matplotlib.axes._subplots.Axes3DSubplot
        The 3D axes object to adjust
    """
    # Get the current limits for each axis
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    # Calculate the range for each axis
    x_range = abs(x_limits[1] - x_limits[0])
    y_range = abs(y_limits[1] - y_limits[0])
    z_range = abs(z_limits[1] - z_limits[0])

    # Find the maximum range to use for all axes
    max_range = max(x_range, y_range, z_range)

    # Calculate the center point for each axis
    x_mid = np.mean(x_limits)
    y_mid = np.mean(y_limits)
    z_mid = np.mean(z_limits)

    # Set new limits with equal range around the center
    # Add 10% padding to ensure all elements are visible
    half_range = max_range / 2 * 1.1
    ax.set_xlim3d([x_mid - half_range, x_mid + half_range])
    ax.set_ylim3d([y_mid - half_range, y_mid + half_range])
    ax.set_zlim3d([z_mid - half_range, z_mid + half_range])


def generate_proper_rotation_matrix():
    """
    Generate a proper rotation matrix using random angles.

    Returns:
    --------
    numpy.ndarray
        3×3 orthogonal rotation matrix with determinant = 1
    """
    # Generate random Euler angles
    angles = np.random.uniform(0, 2 * np.pi, 3)

    # Create rotation matrices for each axis
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(angles[0]), -np.sin(angles[0])],
                   [0, np.sin(angles[0]), np.cos(angles[0])]])

    Ry = np.array([[np.cos(angles[1]), 0, np.sin(angles[1])],
                   [0, 1, 0],
                   [-np.sin(angles[1]), 0, np.cos(angles[1])]])

    Rz = np.array([[np.cos(angles[2]), -np.sin(angles[2]), 0],
                   [np.sin(angles[2]), np.cos(angles[2]), 0],
                   [0, 0, 1]])

    # Combine rotations
    return Rz @ Ry @ Rx


if __name__ == "__main__":
    """
    Example usage and testing of the draw_scene function.
    Tests both 3D and homogeneous coordinate inputs with NEW rotation matrix format.
    """
    print("Testing draw_scene function with NEW rotation matrix format (CAMS×3×3)...")

    # Define scene parameters
    POINTS = 25  # Number of calibration points
    CAMS = 4  # Number of cameras

    # Create a structured calibration object (grid-like pattern)
    print(f"Creating calibration object with {POINTS} points...")
    grid_size = int(np.sqrt(POINTS))
    x_coords = np.linspace(-2, 2, grid_size)
    y_coords = np.linspace(-2, 2, grid_size)
    z_coords = np.zeros(grid_size ** 2)  # Planar calibration object

    # Create grid points
    points_3d = []
    for i, x in enumerate(x_coords):
        for j, y in enumerate(y_coords):
            if len(points_3d) < POINTS:
                points_3d.append([x, y, 0])

    # Test Case 1: 3D coordinates (3×POINTS matrix)
    print("\n=== Test Case 1: 3D Coordinates ===")
    X_3d = np.zeros((3, POINTS))
    for i, point in enumerate(points_3d[:POINTS]):
        X_3d[:, i] = point

    # Test Case 2: Homogeneous coordinates (4×POINTS matrix)
    print("\n=== Test Case 2: Homogeneous Coordinates ===")
    X_homo = np.ones((4, POINTS))
    for i, point in enumerate(points_3d[:POINTS]):
        X_homo[:3, i] = point

    # Create camera centers positioned around the calibration object
    print(f"Creating {CAMS} cameras...")
    camera_distance = 5  # Reduced distance from origin for better visibility
    camera_angles = np.linspace(0, 2 * np.pi, CAMS, endpoint=False)

    C = np.zeros((3, CAMS))
    for i, angle in enumerate(camera_angles):
        C[0, i] = camera_distance * np.cos(angle)  # X coordinate
        C[1, i] = camera_distance * np.sin(angle)  # Y coordinate
        C[2, i] = 2  # Reduced Z coordinate (cameras closer to calibration plane)

    # CHANGE 4: Create rotation matrices in NEW format (CAMS×3×3)
    print("Generating rotation matrices in NEW format (CAMS×3×3)...")
    R = np.zeros((CAMS, 3, 3))  # Changed from (3*CAMS, 3) to (CAMS, 3, 3)
    for i in range(CAMS):
        # Generate a proper rotation matrix
        R_i = generate_proper_rotation_matrix()
        R[i, :, :] = R_i  # Store complete 3×3 matrix in the i-th "slice"

    # Create custom camera IDs
    cams_id = [f"CAM_{i + 1:02d}" for i in range(CAMS)]

    # Test with 3D coordinates
    print("\nDrawing scene with 3D coordinates...")
    fig1 = draw_scene(X_3d, C, R,
                      fig=None,
                      ctype_hint='cloud',
                      scene_title='3D Coordinates - Camera Calibration Scene (NEW Format)',
                      cams_id=cams_id)

    # Test with homogeneous coordinates
    print("\nDrawing scene with homogeneous coordinates...")
    fig2 = draw_scene(X_homo, C, R,
                      fig=None,
                      ctype_hint='cloud',
                      scene_title='Homogeneous Coordinates - Camera Calibration Scene (NEW Format)',
                      cams_id=cams_id)

    # Show both plots
    plt.figure(fig1.number)
    plt.tight_layout()
    plt.figure(fig2.number)
    plt.tight_layout()
    plt.show()

    print("\nBoth test cases completed successfully!")
    print("The function now works with the NEW rotation matrix format (CAMS×3×3)!")
    print(f"Rotation matrix shape: {R.shape}")
    print("Example: R[0] gives you the complete 3×3 rotation matrix for camera 0")