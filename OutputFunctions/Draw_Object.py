import numpy as np
import matplotlib.pyplot as plt
from OutputFunctions.Draw_Cloud import draw_cloud


def draw_cube(X, fig, color='b'):
    """
    Draw a 3D cube by connecting its vertices with lines.

    This function draws the 12 edges of a cube using the provided vertex coordinates.
    The cube is drawn as a wireframe structure in 3D space.

    Parameters:
    -----------
    X : numpy.ndarray
        Shape (3, n) or (4, n) array containing cube corner points.
        If shape is (4, n), only the first 3 rows (x, y, z coordinates) are used.
        Should contain exactly 8 points representing the cube vertices.
    fig : matplotlib.figure.Figure
        Figure object to draw on
    color : str, optional
        Color of the plot lines (default is 'b' for blue)
        Accepts any matplotlib color specification

    Returns:
    --------
    fig : matplotlib.figure.Figure
        The updated figure handle

    Notes:
    ------
    The cube vertices should be ordered appropriately for proper edge connections.
    Standard cube vertex ordering (0-indexed):
        4----5
       /|   /|
      7----6 |
      | 0--|-1
      |/   |/
      3----2
    """
    # Validate input
    if X.shape[1] != 8:
        raise ValueError(f"Cube requires exactly 8 vertices, got {X.shape[1]}")

    # Get current axis or create a new 3D axis if it doesn't exist
    if not fig.axes:
        ax = fig.add_subplot(111, projection='3d')
    else:
        ax = fig.axes[0]
        # Ensure the axis is 3D
        if not hasattr(ax, 'zaxis'):
            fig.clear()
            ax = fig.add_subplot(111, projection='3d')

    # Use only the first 3 rows (x, y, z coordinates)
    points = X[:3, :]

    # Define the 12 edges of the cube
    # Each tuple represents the indices of connected vertices
    edges = [
        # Bottom face (z-low)
        (0, 1), (1, 2), (2, 3), (3, 0),
        # Top face (z-high)
        (4, 5), (5, 6), (6, 7), (7, 4),
        # Vertical edges connecting bottom to top
        (0, 4), (1, 5), (2, 6), (3, 7)
    ]

    # Draw each edge of the cube
    for start_idx, end_idx in edges:
        ax.plot(
            [points[0, start_idx], points[0, end_idx]],  # x coordinates
            [points[1, start_idx], points[1, end_idx]],  # y coordinates
            [points[2, start_idx], points[2, end_idx]],  # z coordinates
            color=color,
            linewidth=1.5
        )

    # Set equal aspect ratio and enable grid for better visualization
    set_axes_equal(ax)
    ax.grid(True, alpha=0.3)

    # Add labels for better understanding
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D Cube Visualization')

    return fig


def draw_octagon(X, fig, color='b'):
    """
    Draw a planar octagon by connecting its vertices in sequence.

    This function creates a closed polygon by connecting 8 vertices in order,
    forming an octagonal shape in 3D space.

    Parameters:
    -----------
    X : numpy.ndarray
        Shape (3, n) or (4, n) array containing octagon corner points.
        If shape is (4, n), only the first 3 rows (x, y, z coordinates) are used.
        Should contain exactly 8 points representing the octagon vertices.
    fig : matplotlib.figure.Figure
        Figure object to draw on
    color : str, optional
        Color of the plot lines (default is 'b' for blue)

    Returns:
    --------
    fig : matplotlib.figure.Figure
        The updated figure handle

    Notes:
    ------
    The octagon vertices should be provided in the order they should be connected.
    The function will connect each vertex to the next one and close the shape
    by connecting the last vertex back to the first.
    """
    # Validate input
    if X.shape[1] != 8:
        raise ValueError(f"Octagon requires exactly 8 vertices, got {X.shape[1]}")

    # Get current axis or create a new 3D axis if it doesn't exist
    if not fig.axes:
        ax = fig.add_subplot(111, projection='3d')
    else:
        ax = fig.axes[0]
        # Ensure the axis is 3D
        if not hasattr(ax, 'zaxis'):
            fig.clear()
            ax = fig.add_subplot(111, projection='3d')

    # Use only the first 3 rows (x, y, z coordinates)
    points = X[:3, :]

    # Draw the octagon by connecting consecutive vertices
    for i in range(8):
        # Connect current vertex to next vertex (wrapping around to 0 for the last vertex)
        next_idx = (i + 1) % 8
        ax.plot(
            [points[0, i], points[0, next_idx]],  # x coordinates
            [points[1, i], points[1, next_idx]],  # y coordinates
            [points[2, i], points[2, next_idx]],  # z coordinates
            color=color,
            linewidth=1.5
        )

    # Optionally, mark the vertices with points for clarity
    ax.scatter(points[0, :], points[1, :], points[2, :],
               color=color, s=50, alpha=0.7)

    # Set equal aspect ratio and enable grid
    set_axes_equal(ax)
    ax.grid(True, alpha=0.3)

    # Add labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D Octagon Visualization')

    return fig


def set_axes_equal(ax):
    """
    Make axes of 3D plot have equal scale for proper proportions.

    This ensures that geometric shapes appear with correct proportions
    (spheres as spheres, cubes as cubes, etc.) rather than being distorted.

    Parameters:
    -----------
    ax : matplotlib.axes.Axes3D
        3D axes object to adjust
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

    # Set equal limits for all axes based on the maximum range
    ax.set_xlim3d([x_mid - max_range / 2, x_mid + max_range / 2])
    ax.set_ylim3d([y_mid - max_range / 2, y_mid + max_range / 2])
    ax.set_zlim3d([z_mid - max_range / 2, z_mid + max_range / 2])


def draw_object(X, ctype, fig=None, color='b'):
    """
    Draw a calibration object of a specific type.

    This is the main function that dispatches to specific drawing functions
    based on the object type. It supports drawing cubes, octagons, and point clouds.

    Parameters:
    -----------
    X : numpy.ndarray
        Shape (3, n) or (4, n) array containing the object's points.
        If shape is (4, n), only the first 3 rows are used.
        - For 'cube': should contain exactly 8 vertices
        - For 'octagon': should contain exactly 8 vertices
        - For 'cloud': can contain any number of points
    ctype : str
        String specifying the type of object to draw.
        Supported types:
        - 'cube': draws a 3D wireframe cube
        - 'octagon': draws a planar octagon
        - 'cloud': draws a point cloud (requires draw_cloud module)
    fig : matplotlib.figure.Figure, optional
        Figure object to draw on. If None, a new figure is created.
    color : str, optional
        Color specification for the plot (default is 'b' for blue)

    Returns:
    --------
    fig : matplotlib.figure.Figure
        The figure handle (newly created or updated)

    Raises:
    -------
    ValueError
        If ctype is not one of the supported object types
    ImportError
        If ctype is 'cloud' but draw_cloud module is not available
    """
    # Input validation
    if not isinstance(X, np.ndarray):
        raise TypeError("X must be a numpy array")

    if X.ndim != 2:
        raise ValueError("X must be a 2D array")

    if X.shape[0] < 3:
        raise ValueError("X must have at least 3 rows (x, y, z coordinates)")

    # Create figure if not provided
    if fig is None:
        fig = plt.figure(figsize=(10, 8))

    # Dispatch to appropriate drawing function based on object type
    if ctype.lower() == 'cube':
        return draw_cube(X, fig, color)
    elif ctype.lower() == 'octagon':
        return draw_octagon(X, fig, color)
    elif ctype.lower() == 'cloud':
        return draw_cloud(X, fig, color)
    else:
        supported_types = ['cube', 'octagon', 'cloud']
        raise ValueError(f"Unknown object type: '{ctype}'. Supported types: {supported_types}")

# from here functions for testing and example


def create_sample_cube(size=1.0, center=(0, 0, 0)):
    """
    Create sample cube vertices for testing purposes.

    Parameters:
    -----------
    size : float, optional
        Side length of the cube (default is 1.0)
    center : tuple, optional
        Center coordinates (x, y, z) of the cube (default is (0, 0, 0))

    Returns:
    --------
    numpy.ndarray
        Shape (3, 8) array containing cube vertices
    """
    cx, cy, cz = center
    half_size = size / 2

    # Define cube vertices relative to center
    vertices = np.array([
        # Bottom face (z = cz - half_size)
        [cx - half_size, cy - half_size, cz - half_size],  # 0
        [cx + half_size, cy - half_size, cz - half_size],  # 1
        [cx + half_size, cy + half_size, cz - half_size],  # 2
        [cx - half_size, cy + half_size, cz - half_size],  # 3
        # Top face (z = cz + half_size)
        [cx - half_size, cy - half_size, cz + half_size],  # 4
        [cx + half_size, cy - half_size, cz + half_size],  # 5
        [cx + half_size, cy + half_size, cz + half_size],  # 6
        [cx - half_size, cy + half_size, cz + half_size],  # 7
    ]).T

    return vertices

# for testing and example


def create_sample_octagon(radius=1.0, center=(0, 0, 0), normal=(0, 0, 1)):
    """
    Create sample octagon vertices for testing purposes.

    Parameters:
    -----------
    radius : float, optional
        Radius of the octagon (default is 1.0)
    center : tuple, optional
        Center coordinates (x, y, z) of the octagon (default is (0, 0, 0))
    normal : tuple, optional
        Normal vector of the octagon plane (default is (0, 0, 1) for xy-plane)

    Returns:
    --------
    numpy.ndarray
        Shape (3, 8) array containing octagon vertices
    """
    cx, cy, cz = center

    # Create octagon vertices in xy-plane first
    angles = np.linspace(0, 2 * np.pi, 9)[:-1]  # 8 points, exclude the duplicate at 2π
    vertices = np.zeros((3, 8))

    vertices[0, :] = cx + radius * np.cos(angles)  # x coordinates
    vertices[1, :] = cy + radius * np.sin(angles)  # y coordinates
    vertices[2, :] = cz  # z coordinates (all same for planar octagon)

    return vertices


if __name__ == "__main__":
    """
    Example usage and testing of the draw_object functions.
    """
    # Create sample data
    cube_data = create_sample_cube(size=2.0, center=(0, 0, 0))
    octagon_data = create_sample_octagon(radius=1.5, center=(0, 0, 0))

    # Example 1: Draw a cube
    print("Drawing cube...")
    fig1 = draw_object(cube_data, 'cube', color='red')
    plt.title('Sample Cube')

    # Example 2: Draw an octagon
    print("Drawing octagon...")
    fig2 = draw_object(octagon_data, 'octagon', color='green')
    plt.title('Sample Octagon')

    # Example 3: Draw random point cloud (if draw_cloud is available)
    print("Drawing point cloud...")
    cloud_data = np.random.rand(3, 100) * 2 - 1  # Random points in [-1, 1]
    fig3 = draw_object(cloud_data, 'cloud', color='blue')
    plt.title('Sample Point Cloud')

    plt.show()