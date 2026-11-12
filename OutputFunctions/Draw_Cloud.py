import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import is_color_like


def draw_cloud(X, fig=None, color='b', marker='o', size=20, alpha=0.7,
               view_angle=None, show_axes=True, title=None):
    """
    Draw a 3D point cloud with enhanced visualization options.

    This function creates a 3D scatter plot of points with proper aspect ratio,
    customizable appearance, and flexible figure management.

    Parameters:
    -----------
    X : numpy.ndarray
        Shape (3, n) or (4, n) array containing 3D points.
        If shape is (4, n), only the first 3 rows (x, y, z coordinates) are used.
        Each column represents one point in 3D space.
    fig : matplotlib.figure.Figure, optional
        Figure object to draw on. If None, a new figure is created.
    color : str or array-like, optional
        Color specification for the points (default is 'b' for blue).
        Can be:
        - Single color string (e.g., 'b', 'red', '#FF0000')
        - Array of colors (one per point)
        - Array of values to be mapped to colors
    marker : str, optional
        Marker style for points (default is 'o' for circles).
        Common options: 'o', 's', '^', 'v', '<', '>', 'D', 'p', '*'
    size : float or array-like, optional
        Size of the markers (default is 20).
        Can be single value or array of sizes (one per point).
    alpha : float, optional
        Transparency level (0.0 to 1.0, default is 0.7).
        0.0 is completely transparent, 1.0 is completely opaque.
    view_angle : tuple, optional
        Viewing angle as (elevation, azimuth) in degrees.
        If None, uses default angle (30, 45) for good 3D perspective.
    show_axes : bool, optional
        Whether to show axis labels and grid (default is True).
    title : str, optional
        Title for the plot. If None, uses default title.

    Returns:
    --------
    fig : matplotlib.figure.Figure
        The figure handle (newly created or updated)

    Raises:
    -------
    ValueError
        If input array has wrong dimensions or invalid parameters
    TypeError
        If input is not a numpy array

    """
    # Input validation
    if not isinstance(X, np.ndarray):
        raise TypeError("X must be a numpy array")

    if X.ndim != 2:
        raise ValueError("X must be a 2D array")

    if X.shape[0] < 3:
        raise ValueError("X must have at least 3 rows (x, y, z coordinates)")

    if X.shape[1] == 0:
        raise ValueError("X must contain at least one point")

    # Validate alpha parameter
    if not (0.0 <= alpha <= 1.0):
        raise ValueError("alpha must be between 0.0 and 1.0")

    # Use only the first 3 rows (x, y, z coordinates)
    points = X[:3, :]
    n_points = points.shape[1]

    # Create figure if not provided
    if fig is None:
        fig = plt.figure(figsize=(10, 8))

    # Get current axis or create a new 3D axis
    if not fig.axes:
        ax = fig.add_subplot(111, projection='3d')
    else:
        ax = fig.axes[0]
        # Ensure the axis is 3D
        if not hasattr(ax, 'zaxis'):
            fig.clear()
            ax = fig.add_subplot(111, projection='3d')

    # Validate and process color parameter
    if isinstance(color, (list, np.ndarray)) and len(color) == n_points:
        # Array of colors or values
        c = color
    elif is_color_like(color):
        # Single color
        c = color
    else:
        # Try to interpret as array of values for colormap
        try:
            c = np.array(color)
            if c.shape[0] != n_points:
                raise ValueError(f"Color array length ({c.shape[0]}) doesn't match number of points ({n_points})")
        except:
            raise ValueError("Invalid color specification")

    # Process size parameter
    if isinstance(size, (list, np.ndarray)):
        if len(size) != n_points:
            raise ValueError(f"Size array length ({len(size)}) doesn't match number of points ({n_points})")
        s = size
    else:
        s = size

    # Create the scatter plot
    scatter = ax.scatter(
        points[0, :],  # x coordinates
        points[1, :],  # y coordinates
        points[2, :],  # z coordinates
        c=c,
        s=s,
        marker=marker,
        alpha=alpha,
        edgecolors='black',
        linewidth=0.5
    )

    # Add colorbar if using array of values for coloring
    if isinstance(c, np.ndarray) and c.ndim == 1 and len(c) == n_points:
        try:
            if not is_color_like(c[0]):  # Check if it's numeric values, not color names
                cbar = plt.colorbar(scatter, ax=ax, shrink=0.8, aspect=20)
                cbar.set_label('Value', rotation=270, labelpad=15)
        except:
            pass  # Skip colorbar if there's an issue

    # Set equal aspect ratio for accurate geometric representation
    set_axes_equal_cloud(ax, points)

    # Set view angle
    if view_angle is None:
        # Default view angle for good 3D perspective
        ax.view_init(elev=30, azim=45)
    else:
        if len(view_angle) != 2:
            raise ValueError("view_angle must be a tuple of (elevation, azimuth)")
        ax.view_init(elev=view_angle[0], azim=view_angle[1])

    # Configure axes appearance
    if show_axes:
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('X', fontsize=12)
        ax.set_ylabel('Y', fontsize=12)
        ax.set_zlabel('Z', fontsize=12)
    else:
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])

    # Set title
    if title is None:
        ax.set_title(f'3D Point Cloud ({n_points} points)', fontsize=14)
    else:
        ax.set_title(title, fontsize=14)

    # Add point statistics as text
    if n_points > 1:
        # Calculate point cloud statistics
        centroid = np.mean(points, axis=1)
        extent = np.max(points, axis=1) - np.min(points, axis=1)

        stats_text = (f'Points: {n_points}\n'
                      f'Centroid: ({centroid[0]:.2f}, {centroid[1]:.2f}, {centroid[2]:.2f})\n'
                      f'Extent: ({extent[0]:.2f}, {extent[1]:.2f}, {extent[2]:.2f})')

        ax.text2D(0.02, 0.98, stats_text, transform=ax.transAxes,
                  fontsize=9, verticalalignment='top',
                  bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    return fig


def set_axes_equal_cloud(ax, points):
    """
    Set equal aspect ratio for 3D point cloud visualization.

    This function ensures that the point cloud appears with correct proportions
    by setting equal scaling on all three axes.

    Parameters:
    -----------
    ax : matplotlib.axes.Axes3D
        3D axes object to adjust
    points : numpy.ndarray
        Shape (3, n) array of 3D points used to determine appropriate limits
    """
    # Calculate the range of the data
    x_min, x_max = np.min(points[0, :]), np.max(points[0, :])
    y_min, y_max = np.min(points[1, :]), np.max(points[1, :])
    z_min, z_max = np.min(points[2, :]), np.max(points[2, :])

    # Add a small margin (5% of range) to prevent points from touching edges
    x_margin = (x_max - x_min) * 0.05
    y_margin = (y_max - y_min) * 0.05
    z_margin = (z_max - z_min) * 0.05

    # Calculate ranges with margins
    x_range = (x_max - x_min) + 2 * x_margin
    y_range = (y_max - y_min) + 2 * y_margin
    z_range = (z_max - z_min) + 2 * z_margin

    # Use the maximum range for all axes to ensure equal scaling
    max_range = max(x_range, y_range, z_range)

    # Calculate centers
    x_center = (x_max + x_min) / 2
    y_center = (y_max + y_min) / 2
    z_center = (z_max + z_min) / 2

    # Set equal limits
    ax.set_xlim3d([x_center - max_range / 2, x_center + max_range / 2])
    ax.set_ylim3d([y_center - max_range / 2, y_center + max_range / 2])
    ax.set_zlim3d([z_center - max_range / 2, z_center + max_range / 2])

# from here functions for testing and example


def create_sample_point_clouds():
    """
    Create various sample point clouds for testing and demonstration.

    Returns:
    --------
    dict
        Dictionary containing different types of sample point clouds
    """
    samples = {}

    # 1. Random uniform distribution
    samples['random'] = np.random.rand(3, 200) * 10 - 5  # Points in [-5, 5]

    # 2. Spherical distribution
    n_sphere = 300
    phi = np.random.uniform(0, 2 * np.pi, n_sphere)
    theta = np.random.uniform(0, np.pi, n_sphere)
    r = np.random.uniform(0.5, 2.0, n_sphere)

    samples['sphere'] = np.array([
        r * np.sin(theta) * np.cos(phi),
        r * np.sin(theta) * np.sin(phi),
        r * np.cos(theta)
    ])

    # 3. Cylindrical distribution
    n_cyl = 250
    angles = np.random.uniform(0, 2 * np.pi, n_cyl)
    heights = np.random.uniform(-3, 3, n_cyl)
    radii = np.random.uniform(0.5, 2.0, n_cyl)

    samples['cylinder'] = np.array([
        radii * np.cos(angles),
        radii * np.sin(angles),
        heights
    ])

    # 4. Helix/spiral
    t = np.linspace(0, 4 * np.pi, 200)
    noise = np.random.normal(0, 0.1, (3, 200))

    samples['helix'] = np.array([
        2 * np.cos(t),
        2 * np.sin(t),
        t
    ]) + noise

    # 5. Gaussian cluster
    n_gauss = 150
    center = np.array([0, 0, 0])
    samples['gaussian'] = np.random.multivariate_normal(
        center,
        np.eye(3) * 2,  # Covariance matrix
        n_gauss
    ).T

    return samples

#for testing and example


def visualize_multiple_clouds(point_clouds, colors=None, save_path=None):
    """
    Visualize multiple point clouds in a single figure with subplots.

    Parameters:
    -----------
    point_clouds : dict
        Dictionary where keys are names and values are 3×n arrays of points
    colors : list, optional
        List of colors for each point cloud. If None, uses default colors.
    save_path : str, optional
        Path to save the figure. If None, doesn't save.

    Returns:
    --------
    fig : matplotlib.figure.Figure
        The figure handle
    """
    n_clouds = len(point_clouds)
    if n_clouds == 0:
        raise ValueError("No point clouds provided")

    # Determine subplot layout
    if n_clouds <= 2:
        rows, cols = 1, n_clouds
    elif n_clouds <= 4:
        rows, cols = 2, 2
    elif n_clouds <= 6:
        rows, cols = 2, 3
    else:
        rows, cols = 3, 3

    # Create figure with subplots
    fig = plt.figure(figsize=(5 * cols, 4 * rows))

    # Default colors if not provided
    if colors is None:
        colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'cyan']

    # Plot each point cloud
    for i, (name, points) in enumerate(point_clouds.items()):
        if i >= rows * cols:
            break  # Skip if too many point clouds

        ax = fig.add_subplot(rows, cols, i + 1, projection='3d')

        color = colors[i % len(colors)]
        scatter = ax.scatter(points[0, :], points[1, :], points[2, :],
                             c=color, alpha=0.7, s=20)

        # Set equal aspect ratio
        set_axes_equal_cloud(ax, points)

        # Configure appearance
        ax.set_title(f'{name.capitalize()} ({points.shape[1]} points)', fontsize=12)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.grid(True, alpha=0.3)
        ax.view_init(elev=30, azim=45)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig


if __name__ == "__main__":
    """
    Example usage and testing of the draw_cloud function.
    """
    print("Testing draw_cloud function with various examples...")

    # Example 1: Basic usage with random points
    print("1. Basic random point cloud...")
    random_points = np.random.rand(3, 100) * 10 - 5
    fig1 = draw_cloud(random_points, color='blue', size=30, alpha=0.8)

    # Example 2: Colored by z-coordinate
    print("2. Point cloud colored by z-coordinate...")
    z_values = random_points[2, :]
    fig2 = draw_cloud(random_points, color=z_values, size=40, alpha=0.9,
                      title='Points Colored by Z-Value')

    # Example 3: Different marker styles and sizes
    print("3. Different marker styles...")
    fig3 = draw_cloud(random_points, color='red', marker='^', size=50,
                      view_angle=(45, 60), title='Triangular Markers')

    # Example 4: Multiple sample point clouds
    print("4. Various geometric point clouds...")
    samples = create_sample_point_clouds()

    # Visualize each sample individually
    for name, points in samples.items():
        print(f"   - {name}: {points.shape[1]} points")
        fig = draw_cloud(points, title=f'{name.capitalize()} Point Cloud')

    # Example 5: Multiple clouds in one figure
    print("5. Multiple point clouds in subplots...")
    fig_multi = visualize_multiple_clouds(samples)

    print("\nAll examples created successfully!")
    print("Call plt.show() to display the figures.")

    # Uncomment the next line to display all figures
    # plt.show()