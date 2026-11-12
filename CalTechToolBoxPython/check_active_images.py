import numpy as np


def check_active_images(n_ima=None, active_images=None):
    """
    Determine which calibration images are active (enabled for processing).

    In the full CalTech toolbox this is a MATLAB script (not a function) that
    runs inside the shared workspace and rebuilds the active_images array.
    The 3-line stub uploaded is a minimal fallback for the single-image case.

    In Python we model it as a utility function used by both comp_error_calib
    and extract_parameters. It validates inputs and also returns ind_active —
    the list of indices of active images — which extract_parameters requires.

    Parameters
    ----------
    n_ima         : int — total number of calibration images (default 1)
    active_images : list/array of int/bool, length n_ima
                    1/True = image is used, 0/False = image is skipped
                    None → all images are marked active

    Returns
    -------
    n_ima         : int
    active_images : list of int (1 = active, 0 = inactive), length n_ima
    ind_active    : list of int — 0-based indices of active images
    """

    if n_ima is None:
        n_ima = 1

    if active_images is None:
        active_images = [1] * n_ima
    else:
        active_images = [int(bool(a)) for a in active_images]
        if len(active_images) != n_ima:
            raise ValueError(
                f"active_images length ({len(active_images)}) "
                f"must match n_ima ({n_ima})."
            )

    # ind_active: 0-based indices of images that are active
    # (mirrors MATLAB's ind_active = find(active_images) — 1-based there)
    ind_active = [i for i, a in enumerate(active_images) if a]

    return n_ima, active_images, ind_active
