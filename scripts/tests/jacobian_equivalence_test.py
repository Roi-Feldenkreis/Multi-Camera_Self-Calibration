"""
jacobian_equivalence_test.py — run from the project root:
    C:\\Users\\User\\anaconda3\\python.exe jacobian_equivalence_test.py

Checks that the masked numerical_jacobian_221 (new) gives exactly the same
columns as the ORIGINAL full 16-column version (embedded below, verbatim),
using your real Multimedia package (Snell_core, Multimedia_params,
Iterative_radial_shift). Also reports the speed-up per call.
"""
import copy
import time
import numpy as np
import sys as _sys, pathlib as _pl; _sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[2]))  # repo root on path
from Multimedia.Multimedia_params import MultimediaParams, CameraParams
from Multimedia.Iterative_radial_shift import (observation_equation_221,
                                               multimedia_radial_shift,
                                               collinearity_condition,
                                               numerical_jacobian_221)


def original_numerical_jacobian_221(particle, cam, x_observed, y_observed, delta=1e-5):
    """The original implementation (full 16 columns, deep copies) — reference."""
    param_names = ['position_0', 'position_1', 'position_2', 'angles_0', 'angles_1', 'angles_2',
                   'c', 'xh', 'yh', 'k1', 'k2', 'k3', 'p1', 'p2', 'a1', 'a2']
    vx0, vy0, _ = observation_equation_221(particle, cam, x_observed, y_observed)
    J = np.zeros((2, 16))
    for col, name in enumerate(param_names):
        cam_plus = copy.deepcopy(cam)
        if name.startswith('position_'):
            idx = int(name[-1]); cam_plus.position = cam.position.copy(); cam_plus.position[idx] += delta
        elif name.startswith('angles_'):
            idx = int(name[-1]); cam_plus.angles = cam.angles.copy(); cam_plus.angles[idx] += delta
        else:
            setattr(cam_plus, name, getattr(cam, name) + delta)
        vx1, vy1, _ = observation_equation_221(particle, cam_plus, x_observed, y_observed)
        J[0, col] = (vx1 - vx0) / delta
        J[1, col] = (vy1 - vy0) / delta
    return J


mm = MultimediaParams(n1=1.0, n2=[1.46], d=[5.0], n3=1.33)
rng = np.random.default_rng(3)
cams = [CameraParams(cam_id=k, position=np.array(p), angles=np.array(a), c=610.0, xh=376.0, yh=240.0, mm=mm)
        for k, (p, a) in enumerate([([-200, 100, 300], [-2.72, -0.28, -1.52]),
                                    ([80, -260, 270], [-2.57, 0.31, -0.12]),
                                    ([150, -50, 290], [-3.01, 0.39, 0.05]),
                                    ([-105, 215, 140], [-2.45, -0.50, -2.20])])]
pts = np.column_stack([rng.uniform(-60, 60, 40), rng.uniform(-60, 60, 40), rng.uniform(-200, -40, 40)])

worst = 0.0
for mask in ([0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 5, 6, 7, 8], None):
    w = 0.0
    for cm in cams:
        for p in pts:
            pb, _, _ = multimedia_radial_shift(p, cm, eps=1e-9, max_iter=300, ocf=1.3)
            x, y = collinearity_condition(pb, cm)
            x += 0.3; y -= 0.2
            Jf = original_numerical_jacobian_221(p, cm, x, y)
            Jm = numerical_jacobian_221(p, cm, x, y, mask=mask)
            w = max(w, float(np.max(np.abs(Jm - (Jf if mask is None else Jf[:, mask])))))
    worst = max(worst, w)
    print(f"mask {str(mask):28s}: max |difference| = {w:.1e}")

p, cm = pts[0], cams[0]
t = time.perf_counter(); [original_numerical_jacobian_221(p, cm, 300.0, 200.0) for _ in range(200)]
t_old = (time.perf_counter() - t) / 200
t = time.perf_counter(); [numerical_jacobian_221(p, cm, 300.0, 200.0, mask=[0, 1, 2, 3, 4, 5]) for _ in range(200)]
t_new = (time.perf_counter() - t) / 200
print(f"\nper observation: original {t_old * 1e3:.2f} ms, masked (6 params) {t_new * 1e3:.2f} ms "
      f"-> {t_old / t_new:.1f}x faster")
print("\nRESULT:", "PASS - identical" if worst == 0.0
      else f"CHECK - max difference {worst:.1e} (send this output back)")
