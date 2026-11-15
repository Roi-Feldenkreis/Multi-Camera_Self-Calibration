"""
Single-LED detection in one image
(Python port of MultiCamSelfCal/FindingPoints/getpoint.m).

Method (unchanged from the MATLAB lineage):
  1. im_thr = |I - avIM| in the LED's colour channel
  2. reject if max(im_thr) < 5 * stdIM at that pixel, or < LEDthr
  3. blob = im_thr > blob_frac * max ; reject if more than one blob
  4. reject if the LED is closer than round(LEDsize/1.2) px to the border
  5. resample the neighbourhood on a grid with step `subpix` (bicubic)
  6. normalized cross-correlation (mycorr2) with a Gaussian PSF,
     sigma = LEDsize/3 ; position = grid node of maximum correlation

Returned position is (x, y) = (column, row), 0-based (NumPy/OpenCV origin).

FIX summary vs. previous version:
  * images read with IMREAD_UNCHANGED as float64 (grey and 16-bit images now work)
  * difference computed in float -> no uint8 wrap-around before abs()
  * BGR channel order respected (red = channel 2, green = channel 1)
  * avIM / stdIM may be 2-D (same channel as the image), as produced by im2imstat
  * real Gaussian PSF (previous kernel was gaussian_filter(ones) = flat)
  * correlation is normalized, via mycorr2 (previous: raw correlate2d)
  * sub-pixel grid built explicitly -> no off-by-half-pixel from resize()
"""
import numpy as np
import cv2
from scipy import ndimage
from numpy.lib.stride_tricks import sliding_window_view

from mycorr2 import mycorr2

_CHANNEL = {"blue": 0, "green": 1, "red": 2}


def read_channel(path, led_color="gray"):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise IOError(f"Cannot read image: {path}")
    if img.ndim == 3:
        if led_color in _CHANNEL:
            img = img[:, :, _CHANNEL[led_color]]
        else:
            img = cv2.cvtColor(img[:, :, :3], cv2.COLOR_BGR2GRAY)
    return img.astype(np.float64)


def _gauss_psf(gsize, sigma):
    ax = np.arange(-gsize, gsize + 1, dtype=np.float64)
    g = np.exp(-0.5 * (ax / sigma) ** 2)
    return np.outer(g, g)


def getfinepos(im_fit, raw_pos, leds, ledsize, step, showfig=False):
    """Sub-pixel refinement: bicubic resampling + NCC with a Gaussian PSF."""
    t = np.arange(-leds, leds + step / 2, step)                  # offsets in px
    rr, cc = np.meshgrid(raw_pos[0] + t, raw_pos[1] + t, indexing="ij")
    zoom = ndimage.map_coordinates(im_fit, [rr, cc], order=3, mode="nearest")

    gsize = int(round(ledsize / 2 / step))
    psf = _gauss_psf(gsize, ledsize / 3 / step)
    k = 2 * gsize + 1
    if zoom.shape[0] <= k:
        raise ValueError("LED neighbourhood smaller than PSF; increase LEDsize/1.2 window")

    win = sliding_window_view(zoom, (k, k))                      # (P, P, k, k)
    P = win.shape[0]
    X = win.reshape(P * P, k * k).T                              # im2col layout
    G = psf.ravel()
    Gn = G - G.mean()
    R = mycorr2(X, G, Gn, np.sqrt(np.sum(Gn ** 2))).reshape(P, P)

    r, c = np.unravel_index(np.argmax(R), R.shape)
    fine = np.array([raw_pos[0] + t[r + gsize], raw_pos[1] + t[c + gsize]])

    if showfig:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2)
        ax[0].imshow(zoom, cmap="gray"); ax[0].set_title("resampled")
        ax[1].imshow(R, cmap="hot"); ax[1].set_title("NCC")
        plt.show()
    return fine


def getpoint(imname, showfig, imconfig, avIM, stdIM, return_reason=False):
    """
    imconfig keys: LEDsize (px), LEDcolor ('green'|'red'|'blue'|'gray'),
                   subpix (grid step in px, e.g. 1/3), LEDthr (default 70),
                   blob_frac (default 0.99, as in the previous port)
    Returns ((x, y), err) or ((x, y), err, reason) with return_reason=True.
    """
    def out(pos, err, why):
        return (pos, err, why) if return_reason else (pos, err)

    ledsize = imconfig["LEDsize"]
    ledthr = imconfig.get("LEDthr", 70)
    step = imconfig["subpix"]
    blob_frac = imconfig.get("blob_frac", 0.99)

    im = read_channel(imname, imconfig.get("LEDcolor", "gray"))
    im_thr = np.abs(im - avIM)

    idx = np.argmax(im_thr)
    peak = im_thr.flat[idx]
    if peak < 5 * stdIM.flat[idx] or peak < ledthr:
        return out((0.0, 0.0), True, "no_led")

    lab, num = ndimage.label(im_thr > blob_frac * peak, structure=np.ones((3, 3)))
    if num > 1:
        return out((0.0, 0.0), True, "multi_blob")
    raw_pos = np.round(ndimage.center_of_mass(lab > 0)).astype(int)

    leds = int(round(ledsize / 1.2))
    rows, cols = im_thr.shape
    if (raw_pos[0] - leds < 0 or raw_pos[0] + leds >= rows or
            raw_pos[1] - leds < 0 or raw_pos[1] + leds >= cols):
        return out((0.0, 0.0), True, "border")

    fine = getfinepos(im, raw_pos, leds, ledsize, step, showfig)
    return out((float(fine[1]), float(fine[0])), False, "ok")
