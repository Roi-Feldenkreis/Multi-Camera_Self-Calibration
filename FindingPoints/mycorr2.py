import numpy as np


def mycorr2(X, G, Gn, Gn2):
    """
    Normalized cross-correlation of every column of X with the template G
    (Python port of MultiCamSelfCal/FindingPoints/mycorr2.m).

    Parameters
    ----------
    X   : (K, P) array, one flattened image patch per column (im2col layout)
    G   : (K,)   flattened template (kept for signature compatibility)
    Gn  : (K,)   G - mean(G), precomputed
    Gn2 : float  sqrt(sum(Gn**2)), precomputed

    Returns
    -------
    R : (P,) correlation coefficient in [-1, 1] for each column of X.
    """
    X = np.asarray(X, dtype=np.float64)
    mXn = X - X.mean(axis=0, keepdims=True)
    smX = np.sqrt(np.sum(mXn ** 2, axis=0))           # FIX: sqrt was missing
    num = Gn @ mXn
    den = smX * Gn2
    with np.errstate(invalid="ignore", divide="ignore"):
        R = np.where(den > 0, num / den, 0.0)
    return R
