"""
Apply_anchor.py
===============

Dispatch wrapper that picks the right anchor at run time:

  - If config.anchor_mode == 'xyz' and a known_camera_positions file
    is available, use the XYZ anchor.
  - If config.anchor_mode == 'z', use the Z-only anchor.
  - Otherwise fall back to the existing apply_metric_anchor.

This is the ONLY edit point needed in GoCal_safe.py.  The two call
sites in GoCal_safe.py change like so:

OLD:
    from CoreFunctions.Metric_anchor import apply_metric_anchor
    ...
    Pe, Xe, Ce, Re, info = apply_metric_anchor(Pe, Xe, Ce, Re, config,
                                               verbose=...)

NEW:
    from CoreFunctions.Apply_anchor import dispatch_anchor
    ...
    Pe, Xe, Ce, Re, info = dispatch_anchor(Pe, Xe, Ce, Re, config,
                                           verbose=...)

The dispatch reads three optional config attributes:
  - config.anchor_mode    : 'metric' (default), 'xyz', or 'z'
  - config.anchor_prior_file : path to JSON file with known camera positions
  - config.anchor_noise_mm   : noise sigma in mm (default 0.0)
  - config.anchor_seed       : RNG seed (default 42)

All four can be set in the config object (or by adding fields to
multicamselfcal.cfg and reading them in CalibrationSystem.py).
For an experiment-by-experiment switch the simplest approach is to
override them at the top of GoCal_safe.py:

    config.anchor_mode = 'xyz'
    config.anchor_prior_file = 'ground_truth.json'
    config.anchor_noise_mm = 10.0
"""
from __future__ import annotations
import json
import os
import numpy as np
from typing import Optional

from CoreFunctions.Metric_anchor import apply_metric_anchor
from CoreFunctions.Camera_anchor import (
    apply_camera_anchor_xyz,
    apply_camera_anchor_z,
)


def _load_known_positions(path: str) -> np.ndarray:
    """Read a ground-truth-style JSON and return a (3, N) array of positions."""
    with open(path, 'r') as f:
        data = json.load(f)
    cams = data['cameras']
    cams_sorted = sorted(cams, key=lambda c: c['id'])
    positions = np.array([c['position_mm'] for c in cams_sorted], dtype=float).T
    return positions   # (3, N)


def dispatch_anchor(Pe, Xe, Ce, Re, config, verbose: bool = True):
    """
    Pick the right anchor based on the config and call it.

    Returns the same 5-tuple as the underlying anchor functions:
      Pe, Xe, Ce, Re, info
    """
    mode = getattr(config, 'anchor_mode', 'metric')

    if mode == 'metric':
        # Standard behaviour — unchanged from existing pipeline.
        return apply_metric_anchor(Pe, Xe, Ce, Re, config, verbose=verbose)

    prior_path = getattr(config, 'anchor_prior_file', None)
    if not prior_path or not os.path.exists(prior_path):
        # Configured for camera anchor but no prior file — warn & fall back.
        if verbose:
            print(f"  WARNING: anchor_mode={mode!r} but no prior file at "
                  f"{prior_path!r}; falling back to metric anchor")
        return apply_metric_anchor(Pe, Xe, Ce, Re, config, verbose=verbose)

    noise_std = float(getattr(config, 'anchor_noise_mm', 0.0))
    seed      = getattr(config, 'anchor_seed', 42)

    try:
        known_positions = _load_known_positions(prior_path)
    except Exception as e:
        if verbose:
            print(f"  WARNING: failed to load {prior_path}: {e}; "
                  f"falling back to metric anchor")
        return apply_metric_anchor(Pe, Xe, Ce, Re, config, verbose=verbose)

    n_recon = Ce.shape[1]
    if known_positions.shape[1] != n_recon:
        if verbose:
            print(f"  WARNING: prior has {known_positions.shape[1]} cameras "
                  f"but reconstruction has {n_recon}; falling back")
        return apply_metric_anchor(Pe, Xe, Ce, Re, config, verbose=verbose)

    if mode == 'xyz':
        return apply_camera_anchor_xyz(
            Pe, Xe, Ce, Re,
            known_positions=known_positions,
            noise_std_mm=noise_std, rng_seed=seed, verbose=verbose,
        )
    elif mode == 'z':
        return apply_camera_anchor_z(
            Pe, Xe, Ce, Re,
            known_z_heights=known_positions[2, :],
            noise_std_mm=noise_std, rng_seed=seed, verbose=verbose,
        )
    else:
        if verbose:
            print(f"  WARNING: unknown anchor_mode={mode!r}; "
                  f"falling back to metric anchor")
        return apply_metric_anchor(Pe, Xe, Ce, Re, config, verbose=verbose)
