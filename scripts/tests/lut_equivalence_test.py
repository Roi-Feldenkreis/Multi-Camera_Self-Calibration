"""
lut_equivalence_test.py — run from the project root:
    C:\\Users\\User\\anaconda3\\python.exe lut_equivalence_test.py

Checks that the vectorised LUT build (MultimediaLUT._solve_ratio_grid) gives the
same R̄/R table as the original scalar solver (MultimediaLUT._solve_R_bar, which
uses your real Snell_core.snell_all_angles), cell by cell, on the 160x160 grid
at three camera heights spanning your rig. Takes ~1 minute (the scalar
reference is the slow part — that is the point of the fix).
"""
import time
import numpy as np
import sys as _sys, pathlib as _pl; _sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[2]))  # repo root on path
from Multimedia.Multimedia_params import MultimediaParams, CameraParams
from Multimedia.Multimedia_lut import MultimediaLUT

mm = MultimediaParams(n1=1.0, n2=[1.46], d=[5.0], n3=1.33)
R_vals = np.linspace(0.01, 376.0, 160)
Zp_vals = np.linspace(0.01, 300.0, 160)
TOL = 1e-9          # max allowed |ratio difference| (~1e-7 mm of shift)

worst = 0.0
for Z0 in (318.1, 406.1, 464.5):
    lut = MultimediaLUT(CameraParams(position=np.array([0.0, 0.0, Z0]), mm=mm))
    t0 = time.perf_counter()
    ref = np.array([[lut._solve_R_bar(R, Z0, mm.t_total, Zp, mm) / R for Zp in Zp_vals]
                    for R in R_vals])
    t_ref = time.perf_counter() - t0
    t0 = time.perf_counter()
    vec = MultimediaLUT._solve_ratio_grid(R_vals, Zp_vals, Z0, mm.t_total, mm)
    t_vec = time.perf_counter() - t0
    d = float(np.max(np.abs(vec - ref)))
    worst = max(worst, d)
    print(f"Z0={Z0:6.1f} mm | max |ratio diff| {d:.2e} "
          f"(max shift diff {float(np.max(np.abs(vec - ref) * R_vals[:, None])):.2e} mm) | "
          f"scalar {t_ref:6.2f} s, vectorised {t_vec * 1000:6.1f} ms -> {t_ref / t_vec:5.0f}x")

print("\nRESULT:", "PASS - identical within tolerance" if worst < TOL
      else f"FAIL - max difference {worst:.2e} exceeds {TOL:.0e}; send this output back")
