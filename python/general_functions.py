import numpy as np
import matplotlib.pyplot as plt
from numba import jit

def normalize_img(img):
    """
    Normalize pixel values of an image to the range [0, 1].

    Parameters:
    - img (numpy.ndarray): Input image as a NumPy array.

    Returns:
    - numpy.ndarray: Normalized image with pixel values in the range [0, 1].

    Example:
    >>> img = np.array([[10, 20, 30], [40, 50, 60]])
    >>> normalize_img(img)
    array([[0. , 0.2, 0.4],
           [0.6, 0.8, 1. ]])
    """
    img_min = np.min(img)
    img_max = np.max(img)
    normalized_img = (img - img_min) / (img_max - img_min)
    return normalized_img

def show_images(imgs, imgs_labels=None, figsize=(20, 10), show_axis=True):
    """
    Display images horizontally using Matplotlib subplots.

    Parameters:
    - imgs: List of images (NumPy arrays) to display.
    - imgs_labels: (Optional) List of labels corresponding to the images.

    Example Usage:
    show_images([image1, image2], ['Label 1', 'Label 2'])
    """
    # Create a figure with subplots based on the number of images
    fig, axes = plt.subplots(1, len(imgs), figsize=figsize)

    # Plot each image on its respective subplot
    for i in range(len(imgs)):
        axes[i].imshow(imgs[i], cmap="gray")  # Display image in grayscale
        if imgs_labels:
            axes[i].set_title(imgs_labels[i])  # Set title if labels provided
        
        if not show_axis:
            axes[i].set_axis_off()

    # Show the plot
    plt.tight_layout()
    plt.show()

@jit(nopython=True)
def sobel_numba(img):
    """
    Apply the Sobel operator to compute the gradient magnitude and orientation of an image.

    Parameters:
    - img (numpy.ndarray): Input grayscale image.

    Returns:
    - tuple: A tuple containing two arrays - the gradient magnitude and orientation in degrees.
    The gradient magnitude represents the strength of edges, and the orientation provides
    the direction of the edges.

    Example:
    ```python
    import numpy as np
    magnitude, orientation = sobel_numba(image)
    ```
    """
    rows, cols = img.shape
    grad_x = np.zeros_like(img)
    grad_y = np.zeros_like(img)

    # Sobel kernels
    kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    kernel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            # Compute gradient in x-direction
            gx = np.sum(img[i-1:i+2, j-1:j+2] * kernel_x)
            # Compute gradient in y-direction
            gy = np.sum(img[i-1:i+2, j-1:j+2] * kernel_y)

            grad_x[i, j] = gx
            grad_y[i, j] = gy

    # Compute magnitude and orientation
    magnitude = np.sqrt(grad_x**2 + grad_y**2)
    orientation = np.arctan2(grad_y, grad_x)
    orientation_deg = np.degrees(orientation)
    
    return magnitude, orientation_deg

@jit(nopython=True)
def compute_weighted_histogram(window, weights, num_bins):
    """
    Compute a weighted histogram from a window of values.

    Parameters:
    - window (numpy.ndarray): 2D array representing the window of values.
    - weights (numpy.ndarray): 2D array of the same shape as the window, representing weights
    associated with each value in the window.
    - num_bins (int): Number of bins for the histogram.

    Returns:
    - numpy.ndarray: 1D array representing the weighted histogram.

    """
    hist = np.array([0 for i in range(num_bins)])
    min_val = -180
    max_val = 180

    bin_edges = np.linspace(min_val, max_val, num_bins + 1)

    for i in range(window.shape[0]):
        for j in range(window.shape[1]):
            value = window[i, j]
            weight = weights[i, j]

            bin_index = np.searchsorted(bin_edges, value, side='right') - 1
            if 0 <= bin_index < num_bins:
                hist[bin_index] += weight
    return hist

@jit(nopython=True)
def get_hist(array, nbins=8):
    """
    Compute weighted histogram of gradient orientations from an input array.

    Parameters:
    - array (numpy.ndarray): Input array for gradient orientation computation.
    - nbins (int): Number of bins for the histogram (default: 8).

    Returns:
    - numpy.ndarray: Weighted histogram of gradient orientations.
    """
    gradient_magnitude, gradient_orientation = sobel_numba(array * 256)
    return compute_weighted_histogram(gradient_orientation, gradient_magnitude, nbins)

