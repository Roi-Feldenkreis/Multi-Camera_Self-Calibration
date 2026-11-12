import numpy as np
import matplotlib.pyplot as plt


def show_img(image_matrix, fig=None, colormap='gray', show_colorbar=False,
             title=None, auto_show=True):
    """
    Display scaled images with customizable options.

    This function provides a flexible way to display 2D numpy arrays as images
    with proper scaling, aspect ratio, and optional enhancements.

    Parameters:
    -----------
    image_matrix : numpy.ndarray
        2D array representing the image to be displayed
    fig : matplotlib.figure.Figure, optional
        Existing figure object to draw on. If None, creates a new figure.
    colormap : str, optional
        Colormap to use for display (default: 'gray')
    show_colorbar : bool, optional
        Whether to display a colorbar showing intensity scale (default: False)
    title : str, optional
        Title for the image plot (default: None)
    auto_show : bool, optional
        Whether to automatically call plt.show() (default: True)

    Returns:
    --------
    fig : matplotlib.figure.Figure
        The figure handle containing the displayed image

    Raises:
    -------
    ValueError
        If image_matrix is not a 2D numpy array
    """
    # Input validation
    if not isinstance(image_matrix, np.ndarray):
        raise ValueError("image_matrix must be a numpy array")

    if image_matrix.ndim != 2:
        raise ValueError("image_matrix must be a 2D array")

    # Create or use existing figure
    if fig is None:
        fig = plt.figure(figsize=(8, 6))
    else:
        plt.figure(fig.number)

    # Clear the current axes to ensure clean display
    plt.clf()

    # Display the image with specified colormap
    img_display = plt.imshow(image_matrix, cmap=colormap, aspect='equal')

    # Set axis properties for optimal image display
    plt.axis('image')  # This combines equal aspect ratio with tight axis limits

    # Add optional enhancements
    if show_colorbar:
        plt.colorbar(img_display, label='Intensity')

    if title:
        plt.title(title)

    # Remove tick marks for cleaner appearance
    plt.xticks([])
    plt.yticks([])

    # Optionally show the plot
    if auto_show:
        plt.show()

    return fig

# This function is for example only
def create_sample_images():
    """
    Create sample images for testing the show_img function.

    Returns:
    --------
    dict : Dictionary containing various sample images
    """

    # Gradient image (smooth transition)
    x = np.linspace(0, 1, 100)
    y = np.linspace(0, 1, 100)
    samples['gradient'] = np.outer(x, y)

    # Circular pattern
    x = np.linspace(-2, 2, 100)
    y = np.linspace(-2, 2, 100)
    X, Y = np.meshgrid(x, y)
    samples['circular'] = np.exp(-(X ** 2 + Y ** 2))

    # Random noise
    samples['noise'] = np.random.rand(100, 100)

    # Checkerboard pattern
    samples['checkerboard'] = np.kron([[1, 0] * 4, [0, 1] * 4] * 4, np.ones((10, 10)))

    return samples


if __name__ == "__main__":
    samples = {}
    # Example usage demonstrating various features
    print("Creating sample images...")
    samples = create_sample_images()

    # Example 1: Basic usage with gradient
    print("\n1. Basic gradient image:")
    fig1 = show_img(samples['gradient'],
                    title='Gradient Image',
                    show_colorbar=True,
                    auto_show=False)

    # Example 2: Circular pattern with different colormap
    print("\n2. Circular pattern with viridis colormap:")
    fig2 = show_img(samples['circular'],
                    colormap='viridis',
                    title='Circular Pattern',
                    show_colorbar=True,
                    auto_show=False)

    # Example 3: Multiple images in subplots
    print("\n3. Multiple images in subplots:")
    fig3 = plt.figure(figsize=(12, 8))

    # Create 2x2 subplot layout
    for i, (name, img) in enumerate(samples.items(), 1):
        plt.subplot(2, 2, i)
        plt.imshow(img, cmap='gray', aspect='equal')
        plt.title(f'{name.title()} Image')
        plt.axis('image')
        plt.xticks([])
        plt.yticks([])

    plt.tight_layout()
    plt.show()

    # Example 4: Using existing figure
    print("\n4. Using existing figure:")
    existing_fig = plt.figure(figsize=(10, 5))
    show_img(samples['checkerboard'],
             fig=existing_fig,
             title='Checkerboard Pattern',
             colormap='hot')