import numpy as np
import cv2
from scikit import measure
from scipy.ndimage import gaussian_filter
from scipy.signal import correlate2d
import matplotlib.pyplot as plt


def getpoint(imname, showfig, imconfig, avIM, stdIM):
    """
    Extracts the position of an LED from an image.

    Parameters:
    imname (str): Name of the image file (full path expected).
    showfig (bool): Flag to show figures for debugging (True -> on, False -> off).
    imconfig (dict): Configuration for image processing, e.g., 'LEDsize', 'LEDthr', 'LEDcolor', 'subpix'.
    avIM (np.array): Average image for reference.
    stdIM (np.array): Image of standard deviations.
    
    Returns:
    pos (tuple): Coordinates of the LED (x, y) or (0, 0) if an error occurs.
    err (bool): Error flag, True if an error occurred.
    """
    
    err = False
    SUB_PIX = 1 / imconfig['subpix']
    LEDSIZE = imconfig['LEDsize']
    LEDTHR = imconfig.get('LEDthr', 70)  # Default threshold is 70 if not specified

    # Read image and check if it's grayscale or RGB
    im_orig = cv2.imread(imname)
    if len(im_orig.shape) == 2:  # Grayscale
        im = {'I': im_orig}
    else:  # Color image
        im = {
            'R': im_orig[:,:,2],  # Red channel
            'G': im_orig[:,:,1],  # Green channel
            'B': im_orig[:,:,0],  # Blue channel
        }

    # Select the correct channel based on LED color
    if imconfig['LEDcolor'] == 'green':
        im_thr = np.abs(im['G'] - avIM[:,:,1]).astype(np.uint8)
        im_std = stdIM[:,:,1]
        im_fit = im['G']
    elif imconfig['LEDcolor'] == 'red':
        im_thr = np.abs(im['R'] - avIM[:,:,0]).astype(np.uint8)
        im_std = stdIM[:,:,0]
        im_fit = im['R']
    else:
        im_thr = np.abs(im['I'] - avIM).astype(np.uint8)
        im_std = stdIM
        im_fit = im['I']

    # Display original and thresholded image if requested
    if showfig:
        plt.figure()
        plt.imshow(im_orig)
        plt.title(f"{imname} original")
        plt.figure()
        plt.imshow(im_thr, cmap='gray')
        plt.title(f"{imname} image to be thresholded")
        plt.show()

    max_intensity = np.max(im_thr)
    idx = np.argmax(im_thr)
    if (im_thr.flat[idx] < 5 * im_std.flat[idx]) or (im_thr.flat[idx] < LEDTHR):
        print(f"No LED in the image. Max intensity: {max_intensity}")
        return (0, 0), True
    
    # Label connected regions (blobs) and find their centroids
    labeled_img, num = measure.label(im_thr > max_intensity * 0.99, return_num=True)
    if num > 1:
        print("More than one blob detected")
        return (0, 0), True

    props = measure.regionprops(labeled_img)
    raw_pos = np.round(props[0].centroid).astype(int)

    # Ensure that the LED is not too close to the border
    leds_size = round(LEDSIZE / 1.2)
    rows, cols = im_thr.shape
    if (raw_pos[0] - leds_size < 0 or raw_pos[0] + leds_size >= rows or
            raw_pos[1] - leds_size < 0 or raw_pos[1] + leds_size >= cols):
        print("LED position too close to image border")
        return (0, 0), True

    # Extract the region of interest around the LED
    led_rows = slice(raw_pos[0] - leds_size, raw_pos[0] + leds_size + 1)
    led_cols = slice(raw_pos[1] - leds_size, raw_pos[1] + leds_size + 1)
    sub_img = im_fit[led_rows, led_cols]

    # Display LED sub-image
    if showfig:
        plt.figure()
        plt.imshow(im_thr > max_intensity * 0.99, cmap='gray')
        plt.plot(raw_pos[1], raw_pos[0], 'g+')
        plt.title('LED Detection')
        plt.show()

    # Refine LED position using Gaussian interpolation
    fine_pos = getfinepos(sub_img, raw_pos, leds_size, LEDSIZE, SUB_PIX, showfig)

    # Display final LED position
    if showfig:
        plt.figure()
        plt.imshow(im_orig)
        plt.plot(fine_pos[1], fine_pos[0], 'r+', markersize=25, linewidth=3)
        plt.title('Final LED Position')
        plt.show()

    return (fine_pos[1], fine_pos[0]), False


def getfinepos(sub_img, raw_pos, leds_size, LEDSIZE, SUB_PIX, showfig=False):
    """
    Refine the LED position using interpolation and correlation with a Gaussian PSF.
    """
    
    Gsize = round(SUB_PIX * LEDSIZE / 2)
    zoomed_img = cv2.resize(sub_img, None, fx=SUB_PIX, fy=SUB_PIX, interpolation=cv2.INTER_CUBIC)

    # Gaussian filter as Point Spread Function (PSF)
    gaussian = gaussian_filter(np.ones((2*Gsize+1, 2*Gsize+1)), sigma=SUB_PIX * LEDSIZE / 3)
    
    # Correlation of the zoomed image with the Gaussian PSF
    corr_img = correlate2d(zoomed_img, gaussian, mode='same')

    # Find the position of maximum correlation
    rmax, cmax = np.unravel_index(np.argmax(corr_img), corr_img.shape)
    fine_pos = raw_pos + (np.array([rmax, cmax]) - np.array(zoomed_img.shape) // 2) / SUB_PIX

    # Plot the interpolated region if requested
    if showfig:
        plt.figure()
        plt.subplot(2, 2, 1)
        plt.imshow(zoomed_img, cmap='gray')
        plt.title('Zoomed Image')
        plt.subplot(2, 2, 2)
        plt.imshow(corr_img, cmap='hot')
        plt.title('Correlation Coefficients')
        plt.show()

    return fine_pos


# How to call this function
# pos, err = getpoint('image.png', True, imconfig, avIM, stdIM)
