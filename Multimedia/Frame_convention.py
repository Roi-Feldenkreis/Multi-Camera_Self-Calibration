"""
Frame_convention.py
===================
The SINGLE authority for the physical reconstruction frame.

Background
----------
The de-refraction pipeline was suffering from multiple independent "frame
owners" that each decided, separately, where the glass plane is, which way is
down, and what scale the reconstruction has:

  * simple_metric_anchor   (percentile glass + fragile two-sided Z-flip)
  * _triangulate_dlt        (re-derived its own frame from the DLT)
  * MultimediaLUT.shift_point  (Zp = -(z+t), its own depth convention)
  * to_physical_frame       (a SECOND full anchor: camera-normal + own scale)
  * _validate_geometry      (its own glass_bottom test)

Because no two agreed, the anchor could report "95% particles below glass"
while the cloud actually fed to the LUT sat +600 mm on the wrong side.

This class fixes that structurally. The frame is established ONCE, by
`from_euclidize`, and every downstream stage READS from it via `to_frame_*`,
`depth_below_glass`, and `is_valid`. Nobody re-derives it.

Canonical frame
---------------
    glass TOP    at z = z_glass_top      (default 0)
    glass BOTTOM at z = z_glass_top - t
    cameras      ABOVE  (z > glass_top)
    particles    BELOW  (z < glass_bottom)

Orientation is taken from the camera->particle direction (robust, no flip
ambiguity). Scale + origin use the single rig constant (camera-glass distance)
with the glass gauged at the top of the particle cloud — the one physical
assumption that closes the 1-DOF the scalar leaves open.
"""

from __future__ import annotations
import numpy as np


class FrameConvention:
    def __init__(self, T: np.ndarray, z_glass_top: float, t: float, scale: float):
        self.T = np.asarray(T, float)          # 4x4 euclidize -> physical mm
        self.z_glass_top = float(z_glass_top)
        self.t = float(t)
        self.glass_bottom = self.z_glass_top - self.t
        self.scale = float(scale)
        self._Tinv = np.linalg.inv(self.T)

    # ------------------------------------------------------------------
    # Establish the frame ONCE
    # ------------------------------------------------------------------
    @classmethod
    def from_euclidize(cls, cam_pos_euc, Xe3_euc, camera_glass_distance, t,
                       z_glass_top: float = 0.0, particle_top_pct: float = 95.0,
                       camera_distances=None, verbose: bool = True,
                       img_radius=None, camera_los=None):
        """
        Build the frame from a euclidize reconstruction + the rig scalar.

        cam_pos_euc : (K,3) camera centres in the euclidize frame.
        Xe3_euc     : (3,N) particle cloud in the euclidize frame.
        camera_glass_distance : mm, camera ring to glass top (the one rig constant).
        t           : glass thickness (mm).
        camera_distances : optional list of (i, j, d_mm) known pairwise camera
            distances. When given, the arbitrary->mm SCALE is gauged from these
            rigid, fixed distances (a stable ruler) instead of the flaky
            particle-cloud top; the cloud top is then used only to place the glass
            origin. Falls back to the cloud-top scale when absent or unusable.

        Returns a FrameConvention, or None if the geometry is degenerate.
        """
        cam_pos_euc = np.asarray(cam_pos_euc, float)
        Xe3 = np.asarray(Xe3_euc, float)
        if cam_pos_euc.shape[1] != 3:
            cam_pos_euc = cam_pos_euc.T
        cam_center = cam_pos_euc.mean(0)
        pt_center = Xe3.mean(1)

        # --- robust orientation: glass normal = cameras -> particles (no flip) ---
        gn = pt_center - cam_center
        n = np.linalg.norm(gn)
        if n < 1e-10:
            return None
        gn /= n
        z_phys = -gn                                   # physical +Z points toward cameras (up)

        a = np.array([1.0, 0.0, 0.0])
        if abs(a @ z_phys) > 0.9:
            a = np.array([0.0, 1.0, 0.0])
        x_phys = a - (a @ z_phys) * z_phys
        x_phys /= np.linalg.norm(x_phys)
        y_phys = np.cross(z_phys, x_phys)
        Rwe = np.column_stack([x_phys, y_phys, z_phys])  # physical axes in euclidize coords
        R = Rwe.T                                         # euclidize -> physical rotation

        # oriented (unscaled) coords, about cam_center
        cam_rot = (cam_pos_euc - cam_center) @ Rwe        # (K,3)
        pts_rot = (Xe3.T - cam_center) @ Rwe              # (N,3)

        # --- camera-glass height: given, or DERIVED from line-of-sight distances ---
        # 'auto' (camera_glass_distance None): each camera's vertical height above the
        # glass is its measured line-of-sight distance times the cosine of the angle
        # between its direction to the cloud centre (approximating its optical axis)
        # and this frame's vertical. Only directions enter, so the result does not
        # depend on the scale or on the height being estimated. Known bias on the test
        # rig: about +35 mm (stable under seed distortion); the seed tolerates at least
        # -20 to +50 mm (run evidence).
        h_derived = None
        if camera_glass_distance is None or (isinstance(camera_glass_distance, str)
                                             and camera_glass_distance.strip().lower() == 'auto'):
            rows = [(int(i), float(dd)) for i, dd in (camera_los or []) if 0 <= int(i) < cam_rot.shape[0]]
            if len(rows) < 3:
                raise ValueError("Camera-Glass-Distance = auto needs line-of-sight distances "
                                 "(LOS-i in [CameraGeometry]) for at least 3 cameras")
            c_rot = np.nanmean(pts_rot, axis=0)
            h_derived = []
            for i, L in rows:
                dvec = c_rot - cam_rot[i]
                h_derived.append(L * abs(dvec[2]) / np.linalg.norm(dvec))
            camera_glass_distance = float(np.mean(h_derived))

        # --- gauge glass at the TOP of the particle cloud (highest physical z) ---
        glass_rot = float(np.percentile(pts_rot[:, 2], particle_top_pct))
        mean_cam_h = float(cam_rot[:, 2].mean() - glass_rot)
        if abs(mean_cam_h) < 1e-9:
            return None

        # --- SCALE: prefer the rigid inter-camera distances (stable ruler) ------
        scale = camera_glass_distance / mean_cam_h        # cloud-top gauge (fallback)
        scale_source = 'camera_glass_distance/cloud-top'
        used_camera_scale = False
        if camera_distances:
            try:
                from seed_gatekeeper import camera_scale_from_distances
                sc = camera_scale_from_distances(cam_pos_euc, camera_distances)
                if np.isfinite(sc['scale']) and sc['scale'] > 0:
                    scale = float(sc['scale'])
                    used_camera_scale = True
                    scale_source = (f"inter-camera distances "
                                    f"(shape err {100*sc['max_rel_err']:.0f}%)")
            except Exception:
                pass   # keep cloud-top fallback

        # --- compose 4x4 T: X_phys = scale*R*(X_euc - cam_center) + origin shift ---
        # HEIGHT / ORIGIN:
        #  * With the camera-based scale (known distances), the constellation size
        #    is now correct, so place the glass using the OTHER rig constant: put
        #    the cameras exactly camera_glass_distance above the glass (glass at
        #    z_glass_top). This makes the two rig constants consistent AND — the
        #    reason it matters — feeds the de-refraction the correct depth-below-
        #    glass for every particle, instead of depths that are offset by the
        #    ~90 mm cloud-top gauge error. Correct depths => correct refraction
        #    correction => a less-warped cloud.
        #  * Without known distances, fall back to the cloud-top gauge (unchanged).
        T = np.eye(4)
        T[:3, :3] = scale * R
        T[:3, 3] = -scale * R @ cam_center
        if used_camera_scale:
            dz = z_glass_top + camera_glass_distance - scale * float(cam_rot[:, 2].mean())
        else:
            dz = z_glass_top - scale * glass_rot          # cloud-top gauge (fallback)
        T[2, 3] += dz

        frame = cls(T, z_glass_top, t, scale)
        frame.camera_glass_distance_used = float(camera_glass_distance)
        frame.camera_glass_distance_derived = h_derived is not None
        if verbose and h_derived is not None:
            print(f"  [frame] camera-glass height DERIVED from the line-of-sight distances: "
                  f"{camera_glass_distance:.1f} mm (per camera "
                  + " ".join(f"{h:.1f}" for h in h_derived) + " mm)")
        if verbose:
            cam_phys = frame.to_frame_cams(cam_pos_euc)
            pts_phys = frame.to_frame_points(Xe3)
            frac = float(np.mean(pts_phys[2, :] < frame.glass_bottom))
            print(f"  [frame] scale={scale:.5f} [{scale_source}]  mean cam height="
                  f"{cam_phys[:, 2].mean() - z_glass_top:.2f} mm  "
                  f"particles below glass={100 * frac:.0f}%")
            # DIAGNOSTIC (log only): below-glass fraction and mean depth by how far
            # from the image centre each particle is seen. Tests whether the pinhole
            # seed's warp grows with field angle. img_radius: (N,) mean distance of
            # each particle's image points from the principal point, in pixels.
            if img_radius is not None:
                try:
                    r = np.asarray(img_radius, float)
                    z = pts_phys[2, :]
                    ok = np.isfinite(r) & np.isfinite(z)
                    if ok.sum() >= 9:
                        order = np.argsort(r[ok]); thirds = np.array_split(order, 3)
                        rr, zz = r[ok], z[ok]
                        parts = []
                        for name, idx in zip(("inner", "middle", "outer"), thirds):
                            parts.append(f"{name} ({rr[idx].min():.0f}-{rr[idx].max():.0f} px) "
                                         f"{100 * np.mean(zz[idx] < frame.glass_bottom):.0f}% "
                                         f"(mean depth {z_glass_top - np.mean(zz[idx]):.0f} mm)")
                        print("  [seed-diag] below glass by image radius: " + " | ".join(parts))
                except Exception as _e:
                    print(f"  [seed-diag] skipped ({type(_e).__name__}: {_e})")
            # DIAGNOSTIC (log only): (1) how far the glass would have to move up for
            # the seed's particles to clear it (95th-percentile height vs the glass
            # bottom minus a 10 mm margin), and (2) the same with the glass plane
            # FITTED from the measured per-camera line-of-sight distances instead of
            # assumed from the camera-centre -> cloud-centre direction.
            if img_radius is not None:
                try:
                    zq = float(np.nanpercentile(pts_phys[2, :], 95))
                    need_cur = max(0.0, zq - (frame.glass_bottom - 10.0))
                    txt = f"clearance needed: current frame {need_cur:.0f} mm"
                    if camera_los:
                        fit = cls._fit_glass_from_los(cam_phys, pts_phys, camera_los, z_glass_top)
                        if fit is not None:
                            n_los, off_los, res_rms = fit
                            s = n_los @ pts_phys - off_los            # height above the fitted glass top
                            need_los = max(0.0, float(np.nanpercentile(s, 95)) - (-t - 10.0))
                            ang = float(np.degrees(np.arccos(np.clip(n_los[2], -1.0, 1.0))))
                            txt += (f" | line-of-sight frame {need_los:.0f} mm ; line-of-sight glass "
                                    f"normal differs from the current one by {ang:.1f} deg "
                                    f"(fit residual {res_rms:.1f} mm)")
                    print("  [seed-diag] " + txt)
                except Exception as _e:
                    print(f"  [seed-diag] clearance skipped ({type(_e).__name__}: {_e})")
        return frame

    @staticmethod
    def _fit_glass_from_los(cam_phys, pts_phys, camera_los, z_glass_top=0.0):
        """Glass plane fitted from measured line-of-sight distances (diagnostic).
        Each camera's optical axis is approximated by its direction to the cloud
        centre; its vertical height above the glass is then LOS * cos(angle of that
        direction from the glass normal). Unknowns: glass normal n (2 DOF, n_z > 0)
        and plane offset; 4 cameras -> 4 equations:
            C_i . n - LOS_i * (-d_i . n) - off = 0
        Returns (n, off, residual_rms_mm) in the current frame, or None."""
        from scipy.optimize import least_squares
        C = np.asarray(cam_phys, float)
        if C.shape[0] == 3 and C.shape[1] != 3:
            C = C.T
        P = np.asarray(pts_phys, float)
        c = np.nanmean(P, axis=1)
        rows = [(int(i), float(d)) for i, d in camera_los if 0 <= int(i) < C.shape[0]]
        if len(rows) < 3:
            return None
        D = np.array([(c - C[i]) / np.linalg.norm(c - C[i]) for i, _ in rows])
        Ci = np.array([C[i] for i, _ in rows]); L = np.array([d for _, d in rows])

        def unit(a, b):
            v = np.array([a, b, 1.0]); return v / np.linalg.norm(v)

        def res(x):
            n = unit(x[0], x[1])
            return Ci @ n - L * (-(D @ n)) - x[2]

        sol = least_squares(res, x0=[0.0, 0.0, z_glass_top], method='lm')
        n = unit(sol.x[0], sol.x[1])
        return n, float(sol.x[2]), float(np.sqrt(np.mean(sol.fun ** 2)))

    # ------------------------------------------------------------------
    # Consumers READ the frame through these — never re-derive
    # ------------------------------------------------------------------
    def to_frame_points(self, Xe3_euc: np.ndarray) -> np.ndarray:
        """(3,N) euclidize -> (3,N) physical mm."""
        X = np.asarray(Xe3_euc, float)
        return self.T[:3, :3] @ X + self.T[:3, 3:4]

    def to_frame_cams(self, cam_pos_euc: np.ndarray) -> np.ndarray:
        """(K,3) euclidize -> (K,3) physical mm."""
        C = np.asarray(cam_pos_euc, float)
        if C.shape[1] != 3:
            C = C.T
        return (self.T[:3, :3] @ C.T + self.T[:3, 3:4]).T

    def to_frame_Pe(self, Pe: np.ndarray) -> np.ndarray:
        """Map pixel projection matrices into the physical frame (proj-invariant)."""
        Pe = np.asarray(Pe, float)
        out = np.zeros_like(Pe)
        for k in range(Pe.shape[0] // 3):
            out[3 * k:3 * k + 3, :] = Pe[3 * k:3 * k + 3, :] @ self._Tinv
        return out

    def depth_below_glass(self, X) -> np.ndarray:
        """
        Zp = depth below the glass BOTTOM (mm), clamped >= 0.
        Single definition replacing shift_point's Zp = -(z+t).
        X may be (3,) or (3,N).
        """
        X = np.asarray(X, float)
        z = X[2] if X.ndim == 1 else X[2, :]
        Zp = self.glass_bottom - z
        return np.maximum(Zp, 0.0)

    def is_valid(self, cam_pos_phys, Xe3_phys,
                 min_frac_below: float = 0.90, max_cams_below: int = 0,
                 above_glass_margin_mm: float = 10.0):
        """
        The ONE geometry check: cameras above glass top, particles below bottom.
        Inputs must already be in the physical frame (via to_frame_*).

        Tolerant of small human/gauge error: a particle up to
        `above_glass_margin_mm` above the glass bottom still counts as below, and
        a strong majority (>= min_frac_below) is required rather than all.
        Returns (ok: bool, info: dict).
        """
        C = np.asarray(cam_pos_phys, float)
        if C.ndim == 2 and C.shape[1] != 3:
            C = C.T
        X = np.asarray(Xe3_phys, float)
        cam_z = C[:, 2]
        # cameras: allow the same small margin below the glass top before failing
        n_cams_below = int(np.sum(cam_z <= self.z_glass_top - above_glass_margin_mm))
        # particles: forgiving threshold (glass bottom + margin)
        thresh = self.glass_bottom + above_glass_margin_mm
        frac_below = float(np.mean(X[2, :] < thresh))
        ok = (n_cams_below <= max_cams_below) and (frac_below >= min_frac_below)
        info = {'n_cams_below_glass': n_cams_below,
                'frac_particles_below_glass': frac_below,
                'cam_heights_mm': (cam_z - self.z_glass_top).tolist()}
        return ok, info
