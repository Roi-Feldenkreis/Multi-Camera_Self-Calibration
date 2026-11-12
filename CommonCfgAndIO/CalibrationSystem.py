"""
Simplified Multi-Camera Self-Calibration System
================================================

A clean, focused calibration system for .cfg files with:
- One main class (CalibrationSystem)
- Modern OOP CalibrationConfig and CalibrationData
- Primary focus on .cfg file format
- Support for JSON export when needed
- Simple, straightforward API

Usage:
    from calibration_system import CalibrationSystem

    system = CalibrationSystem()
    config = system.load_config('path/to/config.cfg')
    data = system.load_data(config)
    system.save_config(config, 'output.cfg')
"""

import os
import json
import logging
import configparser
import re
import numpy as np
from pathlib import Path
from typing import List, Optional, Union, Tuple, Dict
from dataclasses import dataclass, field

from Multimedia.Multimedia_params import MultimediaParams

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Fixed weight (px per mm) for the inter-camera distance penalty in the multimedia
# bundle adjustment. NOT user-configurable: the value follows from balancing the
# six-term penalty against the ~2*N_obs reprojection residuals (see thesis 3.3.7),
# and exposing it only invites mis-tuning. Any 'Weight' in the config is ignored.
CAMERA_DISTANCE_WEIGHT = 40.0


@dataclass
class CalibrationConfig:
    """Modern OOP configuration structure for multi-camera calibration."""

    # Paths
    data_path: str = ""
    img_path: str = ""

    # Files
    basename: str = ""
    img_names: str = ""
    img_extension: str = "jpg"
    num_cameras: int = 4
    num_projectors: int = 0

    # Images
    led_size: int = 5
    led_color: str = "red"
    led_threshold: int = 30
    subpix: float = 0.5
    resolution: Tuple[int, int] = (640, 480)
    projector_resolution: Tuple[int, int] = (1024, 768)

    # Calibration
    nonlinear_parameters: List[float] = field(default_factory=lambda: [50.0, 0.0, 1.0, 0.0, 0.0, 0.0])
    nonlinear_update: List[int] = field(default_factory=lambda: [1, 0, 1, 0, 0, 0])
    do_global_iterations: bool = False
    global_iteration_max: int = 5
    global_iteration_threshold: float = 0.5
    initial_tolerance: float = 10.0
    num_cameras_fill: int = 4
    do_bundle_adjustment: bool = True
    undo_radial: bool = False
    min_points_value: int = 30
    n_tuples: int = 3
    square_pixels: bool = True
    use_nth_frame: int = 1
    align_existing: str = ''  # empty = disabled; filename = run alignment
    principal_point = np.zeros((num_cameras, 2))
    cameras_2_use: list[int] = field(default_factory=lambda: [])

    # -----------------------------------------------------------------------
    # Multimedia geometry parameters (Maas 1996)
    # -----------------------------------------------------------------------
    # These fields store the refractive-index stack for the
    # air → glass → water optical path that every ray travels in a
    # through-glass PTV experiment.
    #
    # Convention (matches Maas 1996 and liboptv / OpenPTV):
    #   n1  = refractive index of the CAMERA-SIDE medium  (air   ≈ 1.0)
    #   n2  = refractive indices of each GLASS layer       (glass ≈ 1.5)
    #   d   = thickness of each glass layer in mm
    #   n3  = refractive index of the OBJECT-SIDE medium  (water ≈ 1.333)
    #   z_glass_top = Z-coordinate of the glass top surface in mm
    #                 (defines where the air/glass interface sits in world coords)
    #
    # TO DISABLE (air-only experiment, no through-glass correction):
    #   Leave all at their defaults: n1=1.0, n2=[1.0], d=[0.0], n3=1.0
    #   The use_multimedia property returns False → every multimedia call
    #   is skipped automatically.  Existing .cfg files need no changes.
    n1:          float       = 1.0
    n2:          List[float] = field(default_factory=lambda: [1.0])
    d:           List[float] = field(default_factory=lambda: [0.0])
    n3:          float       = 1.0
    z_glass_top: float             = 0.0  # Z of air/glass interface (mm)
    # Camera-Glass-Distance: auto -> derived from the line-of-sight distances in every
    # seed frame (camera_glass_distance_mm then stays 0.0 and is not used).
    camera_glass_auto: bool = False
    camera_glass_distance_mm: float = 0.0  # camera-to-glass distance (mm); enables multimedia BA without ground truth
    # Optional soft constraint: known pairwise camera-to-camera distances (mm).
    # Resolves the scale ambiguity that reprojection alone cannot (see the
    # multimedia BA). Format: list of (i, j, d_ij). Hand-measured => a few mm off,
    # so it is applied as a SOFT penalty. The weight is fixed (CAMERA_DISTANCE_WEIGHT),
    # not user-selectable, because the balance argument (penalty vs. the far more
    # numerous reprojection residuals) sets its value; exposing it only invites
    # mis-tuning.
    camera_distances: List[tuple] = field(default_factory=list)   # [(i, j, d_mm), ...]
    camera_distance_weight: float = 40.0   # px per mm — FIXED (see CAMERA_DISTANCE_WEIGHT)
    # Cyclic order of the cameras around the rig (0-based indices). The six scalar
    # distances fix the camera constellation only up to a REFLECTION; the cyclic
    # order encodes the rig's handedness and breaks that two-fold ambiguity, so a
    # mirror-image reconstruction (which has identical distances and reprojection)
    # can be detected and rejected. Empty => no chirality check.
    camera_cyclic_order: List[int] = field(default_factory=list)
    # Hand-measured distance from each camera, along its line of sight, to the
    # glass top surface (mm), 0-based camera index: [(i, d_mm)]. Used as an
    # in-situ VALIDITY CHECK on the returned rig (seed_gatekeeper.los_check), not
    # as a penalty: it detects a rig at the wrong height/tilt relative to the glass.
    camera_los: List[tuple] = field(default_factory=list)
    # Refraction LUT (Maas correction) grid resolution. The de-refraction stores
    # the ray-bending correction on a grid of n_R (radial) x n_Zp (depth) samples
    # and interpolates between them; too coarse a grid adds a systematic
    # interpolation error that sets a floor on reprojection accuracy. Finer grids
    # cost memory/build time (a one-time cost per calibration) but reduce that
    # floor. Sweep these to find where the reprojection floor stops dropping.
    lut_n_r: int = 80            # radial samples
    lut_n_zp: int = 80           # depth samples
    lut_zp_max: float = 300.0    # max depth below glass covered by the LUT (mm)
    # max radial distance from a camera's nadir covered by the LUT (mm).
    # 0 = legacy behaviour: half the image width (a pixel count used as mm).
    lut_r_max: float = 0.0
    # Camera parameters the multimedia bundle adjustment optimises, as indices into
    # [X0 Y0 Z0 omega phi kappa c xh yh k1 k2 k3 p1 p2 a1 a2]. Default: the 6
    # exterior parameters. Only 0-8 are allowed: distortion (k1..p2) and A/D (a1, a2)
    # are corrected on the observations before the BA, so the projection never uses
    # them and their Jacobian columns would be all zero.
    ba_cam_params: List[int] = field(default_factory=lambda: [0, 1, 2, 3, 4, 5])
    # How the multimedia BA refines an accepted seed:
    #   'alternate'         camera BA (points frozen), then cloud re-triangulation
    #                       (cameras frozen); polish alternates the same way x3
    #   'joint'             one joint camera + point BA; polish is one joint BA
    #   'camera-then-joint' camera BA first (to move away from a warped seed),
    #                       then joint; polish is one joint BA
    ba_mode: str = 'alternate'
    # Seed-stage particle selection (stage-1 test of the wide-coverage problem):
    #   seed_max_radius  : keep only particles seen within this many pixels of the
    #                      image centre in EVERY camera; 0 = off. In this stage it
    #                      applies to the whole run (seed and bundle adjustment).
    #   seed_subset_frac : each seed attempt uses a random fraction of the RANSAC
    #                      inliers, so retries differ even when RANSAC keeps every
    #                      point; 1.0 = off (all inliers, previous behaviour).
    seed_max_radius: float = 0.0
    seed_subset_frac: float = 1.0
    # Stage 2: with seed_max_radius > 0, use the central particles for the SEED only
    # and hand ALL particles to the bundle adjustment after the camera BA (mode
    # 'camera-then-joint'). False = stage-1 behaviour (central particles throughout).
    ba_all_particles: bool = False
    # De-refraction divergence guard: a pass whose largest correction exceeds this
    # many pixels is discarded and the loop stops with the last bounded pass. 0 = off.
    derefract_max_shift_px: float = 0.0
    # Stage-2 handover only from a central joint BA below this reprojection error.
    # Evidence: every rescued central solution was at <= 0.61 px, every failed one
    # at >= 1.2 px (shape errors of 16-107 mm). Default 1 px.
    handover_max_reproj_px: float = 1.0
    # Staged distance weight. The search stages (camera BA, central joint BA) keep
    # CAMERA_DISTANCE_WEIGHT; the final stages (handover joint BAs, polish) use
    # dist_weight_final. Realism evidence: with hand-measured distances (+-4 mm),
    # 40 px/mm moved the optimum ~40 mm in height; ~0.5 px/mm let the images decide.
    # A lighter final weight also switches selection to reprojection and bounds the
    # misfit by dist_misfit_max (early stop, verdict, polish). Defaults = unchanged.
    dist_weight_final: float = CAMERA_DISTANCE_WEIGHT
    dist_misfit_max: float = 4.0
    # Intrinsics: 'rad' = from the .rad files (previous behaviour); 'estimate' = no
    # .rad values used: focal length from the seed's metric upgrade (or Focal-Start-Px),
    # principal point at the image centre; both FREED in the final stages, with a soft
    # principal-point prior (PP-Prior-Sigma-Px). Lens distortion assumed zero.
    intrinsics_mode: str = 'rad'
    focal_start_px: float = 0.0
    pp_prior_sigma_px: float = 8.0
    # Line of sight as a WEAK constraint in the final stages (both modes): the measured
    # distances with this uncertainty in mm; 0 = off (check only, previous behaviour).
    los_sigma_final_mm: float = 0.0

    @property
    def use_multimedia(self) -> bool:
        """
        Returns True when the refractive-index stack is non-trivial,
        i.e. at least one medium differs from air (n=1.0).

        This is the SINGLE gate used throughout the pipeline:

            if config.use_multimedia:
                mm     = config.to_multimedia_params()
                ...    # call Iterative_radial_shift  / Forward_ray_tracing / multimedia_lut
            else:
                ...    # use standard pinhole model — no changes needed

        The condition mirrors the liboptv check in ray_tracing.c:
            if (n1 != 1 || n2[0] != 1 || n3 != 1) -> init_mmLUT()
        """
        return (
            abs(self.n1 - 1.0) > 1e-9
            or any(abs(n - 1.0) > 1e-9 for n in self.n2)
            or abs(self.n3 - 1.0) > 1e-9
        )

    def to_multimedia_params(self) -> 'MultimediaParams':
        """
        Build a MultimediaParams object from this config's n1/n2/d/n3 fields.

        Call this ONCE after load_config(), then pass the result into
        CameraParams(mm=...) for every camera in the system.

        """
        return MultimediaParams(
            n1=self.n1,
            n2=list(self.n2),
            d=list(self.d),
            n3=self.n3,)

    def __post_init__(self):
        """Normalize paths after initialization."""
        # Use os.path.normpath to convert '/' to '\' on Windows, then
        # ensure a trailing separator so f"{data_path}file" always works.
        # However, best practice (and what the rest of the code now does)
        # is to use os.path.join() at every call site — this block is a
        # safety net for any remaining f-string concatenations.
        if self.data_path:
            self.data_path = os.path.normpath(self.data_path) + os.sep
        if self.img_path:
            self.img_path = os.path.normpath(self.img_path) + os.sep


@dataclass
class CalibrationData:
    """Modern OOP data structure for calibration data."""
    Ws: np.ndarray
    id_matrix: np.ndarray
    resolutions: np.ndarray
    projection_matrices: Optional[List[np.ndarray]] = None
    camera_parameters: Optional[List[Dict]] = None
    INT_CAM_PARAMS:bool = False

    def __post_init__(self):
        """Validate data dimensions."""
        if self.Ws.ndim != 2 or self.id_matrix.ndim != 2:
            raise ValueError("Points and ID matrix must be 2D arrays")

        num_cameras = self.id_matrix.shape[0]
        expected_rows = num_cameras * 3

        if self.Ws.shape[0] != expected_rows:
            raise ValueError(f"Points dimension mismatch: expected {expected_rows}, got {self.Ws.shape[0]}")


class CalibrationSystem:
    """
    Main unified calibration system class.
    This is the ONLY class you need to interact with.
    """

    def __init__(self):
        """Initialize the calibration system."""
        self.current_config = None
        self.current_data = None
        logger.info("CalibrationSystem initialized")

    def load_config(self, filepath: Union[str, Path]) -> CalibrationConfig:
        """
        Load configuration from .cfg or .json file.

        Example:
            config = system.load_config('experiment.cfg')
        """
        filepath = Path(filepath)

        if not filepath.exists():
            raise FileNotFoundError(f"Configuration file not found: {filepath}")

        if filepath.suffix == '.cfg':
            config = self._load_from_cfg(filepath)
        elif filepath.suffix == '.json':
            config = self._load_from_json(filepath)
        else:
            raise ValueError(f"Unsupported file format: {filepath.suffix}. Use .cfg or .json")
        if not config.cameras_2_use:
            config.cameras_2_use = list(range(1,config.num_cameras+1))

        self.current_config = config
        logger.info(f"Loaded config: {config.basename}")
        return config

    @staticmethod
    def _parse_align_existing_static(raw: str) -> str:
        """Convert raw cfg value to filename string or empty string."""
        raw = raw.strip()
        if raw.lower() in ('0', 'false', 'no', 'off', 'none', ''):
            return ''
        return raw

    def _load_from_cfg(self, filepath: Path) -> CalibrationConfig:
        """Load configuration from .cfg file (INI format)."""
        parser = configparser.ConfigParser()
        parser.read(filepath)

        def get_str(section, key, default=''):
            return parser.get(section, key, fallback=default)

        def get_int(section, key, default=0):
            try:
                return parser.getint(section, key, fallback=default)
            except ValueError:
                return default

        def get_float(section, key, default=0.0):
            try:
                return parser.getfloat(section, key, fallback=default)
            except ValueError:
                return default

        def get_bool(section, key, default=False):
            value = parser.get(section, key, fallback=str(int(default)))
            return value.lower() in ['1', 'true', 'yes', 'on']

        def get_list(section, key, dtype=int, default=None):
            value = get_str(section, key, '')
            if not value:
                return default if default is not None else []
            return [dtype(x) for x in value.split()]

        def _parse_align_existing(raw: str) -> str:
            raw = raw.strip()
            if raw.lower() in ('0', 'false', 'no', 'off', 'none', ''):
                return ''
            return raw

        config = CalibrationConfig(
            data_path=get_str('PATHS', 'Data-Path', ''),
            img_path=get_str('PATHS', 'Image-Path', ''),
            basename=get_str('Files', 'Basename', ''),
            img_names=get_str('Files', 'Image-Names', ''),
            img_extension=get_str('Files', 'Image-Extension', 'jpg'),
            num_cameras=get_int('Calibration', 'Num-Cameras', 4),
            num_projectors=get_int('Calibration', 'Num-Projectors', 0),
            led_size=get_int('Images', 'LED-Size', 5),
            led_color=get_str('Images', 'LED-Color', 'red'),
            led_threshold=get_int('Images', 'LED-Threshold', 30),
            subpix=get_float('Images', 'Subpix', 0.5),
            resolution=tuple(get_list('Images', 'Resolution', int, [640, 480])),
            projector_resolution=tuple(get_list('Images', 'Projector-Resolution', int, [1024, 768])),
            nonlinear_parameters=get_list('Calibration', 'Nonlinear-Parameters', float,
                                          [50.0, 0.0, 1.0, 0.0, 0.0, 0.0]),
            nonlinear_update=get_list('Calibration', 'Nonlinear-Update', int, [1, 0, 1, 0, 0, 0]),
            do_global_iterations=get_bool('Calibration', 'Do-Global-Iterations', False),
            global_iteration_max=get_int('Calibration', 'Global-Iteration-Max', 5),
            global_iteration_threshold=get_float('Calibration', 'Global-Iteration-Threshold', 0.5),
            initial_tolerance=get_float('Calibration', 'Initial-Tolerance', 10.0),
            num_cameras_fill=get_int('Calibration', 'Num-Cameras-Fill', 4),
            do_bundle_adjustment=get_bool('Calibration', 'Do-Bundle-Adjustment', True),
            undo_radial=get_bool('Calibration', 'Undo-Radial', False),
            min_points_value=get_int('Calibration', 'Min-Points-Value', 30),
            n_tuples=get_int('Calibration', 'N-Tuples', 3),
            square_pixels=get_bool('Calibration', 'Square-Pixels', True),
            use_nth_frame=get_int('Calibration', 'Use-Nth-Frame', 1),
            align_existing=_parse_align_existing(
                parser.get('Calibration', 'Align-Existing', fallback='0')),

        )
        config.principal_point = np.array(get_list('Calibration', 'Principal Point', int, np.zeros((config.num_cameras, 2))))
        if (config.principal_point > 0).all():
            config.principal_point = np.tile(config.principal_point, config.num_cameras)
            config.principal_point = config.principal_point.reshape(config.num_cameras, 2)
        config.cameras_2_use = tuple(get_list('Calibration', 'Cameras_to_use', int, []))
        # -------------------------------------------------------------------
        # Read optional [Multimedia] section
        # -------------------------------------------------------------------
        # If the section is present, inject its values into the config object.
        # If it is absent, the defaults (n1=1.0, n2=[1.0], d=[0.0], n3=1.0)
        # remain and config.use_multimedia == False — zero impact on existing
        # air-only experiments.
        if parser.has_section('Multimedia'):
            mm = self._read_multimedia_from_cfg(parser)
            config.n1          = mm['n1']
            config.n2          = mm['n2']
            config.d           = mm['d']
            config.n3          = mm['n3']
            config.z_glass_top = mm['z_glass_top']
            config.camera_glass_distance_mm = mm['camera_glass_distance_mm']
            config.camera_glass_auto = mm['camera_glass_auto']
            config.lut_n_r     = mm['lut_n_r']
            config.lut_n_zp    = mm['lut_n_zp']
            config.lut_zp_max  = mm['lut_zp_max']
            config.lut_r_max   = mm['lut_r_max']
            config.ba_cam_params = mm['ba_cam_params']
            config.ba_mode = mm['ba_mode']
            config.seed_max_radius = mm['seed_max_radius']
            config.seed_subset_frac = mm['seed_subset_frac']
            config.ba_all_particles = mm['ba_all_particles']
            config.derefract_max_shift_px = mm['derefract_max_shift_px']
            config.handover_max_reproj_px = mm['handover_max_reproj_px']
            config.dist_weight_final = mm['dist_weight_final']
            config.dist_misfit_max = mm['dist_misfit_max']
            config.intrinsics_mode = mm['intrinsics_mode']
            config.focal_start_px = mm['focal_start_px']
            config.pp_prior_sigma_px = mm['pp_prior_sigma_px']
            config.los_sigma_final_mm = mm['los_sigma_final_mm']
            if config.use_multimedia:
                logger.info(
                    f"Multimedia geometry ENABLED: "
                    f"n1={config.n1}, n2={config.n2}, "
                    f"d={config.d} mm, n3={config.n3}"
                )
                logger.info(
                    f"Refraction LUT grid: n_R={config.lut_n_r} x n_Zp={config.lut_n_zp}, "
                    f"Zp_max={config.lut_zp_max:g} mm, "
                    + (f"R_max={config.lut_r_max:g} mm" if config.lut_r_max > 0
                       else "R_max=legacy (half image width)")
                )
                logger.info(
                    "Multimedia BA optimises: "
                    + " ".join(self._BA_PARAM_NAMES[i] for i in config.ba_cam_params)
                    + f"  |  BA-Mode: {config.ba_mode}"
                )
                if abs(config.dist_weight_final - CAMERA_DISTANCE_WEIGHT) > 1e-9 or abs(config.dist_misfit_max - 4.0) > 1e-9:
                    logger.info(f"Distance weight STAGED: {CAMERA_DISTANCE_WEIGHT:g} px/mm for the search (camera BA, "
                                f"central joint BA), {config.dist_weight_final:g} px/mm for the final stages (handover "
                                f"joint BAs, polish); misfit bound {config.dist_misfit_max:g} mm"
                                + ("; selection by reprojection" if config.dist_weight_final < CAMERA_DISTANCE_WEIGHT else ""))
                if config.intrinsics_mode == 'estimate':
                    logger.info("Intrinsics: ESTIMATE - no .rad values used; focal length from "
                                + (f"Focal-Start-Px={config.focal_start_px:g}" if config.focal_start_px > 0
                                   else "the seed's metric upgrade")
                                + f", principal point at the image centre (prior sigma {config.pp_prior_sigma_px:g} px); "
                                "both freed in the final stages; lens distortion assumed zero")
                if config.los_sigma_final_mm > 0:
                    logger.info(f"Line of sight as a WEAK CONSTRAINT in the final stages: sigma "
                                f"{config.los_sigma_final_mm:g} mm (still also a validity check)")
                if config.derefract_max_shift_px > 0:
                    logger.info(f"De-refraction divergence guard ENABLED: passes with corrections above "
                                f"{config.derefract_max_shift_px:g} px are discarded")
                if config.seed_max_radius > 0 or config.seed_subset_frac < 1.0:
                    logger.info(
                        "Seed selection: "
                        + (f"particles within {config.seed_max_radius:g} px of the image centre "
                           f"in every camera "
                           + ("(seed only; the bundle adjustment gets ALL particles)"
                              if config.ba_all_particles else "(whole run)")
                           if config.seed_max_radius > 0 else "all particles")
                        + f"  |  random {100*config.seed_subset_frac:.0f}% of the inliers per seed attempt"
                    )
        else:
            logger.info("No [Multimedia] section found — standard pinhole model.")

        # -------------------------------------------------------------------
        # Read optional [CameraGeometry] section
        # -------------------------------------------------------------------
        # Known pairwise camera-to-camera distances (mm) + cyclic camera order,
        # used as a soft constraint in the multimedia bundle adjustment to fix the
        # scale ambiguity and break the reflection ambiguity. Absent section =>
        # empty => no constraint (unchanged behaviour). The penalty Weight is FIXED
        # in code and is NOT read from the file. Example:
        #   [CameraGeometry]
        #   CyclicOrder = 0,1,2,3
        #   Dist-0-1 = 152.3
        #   Dist-0-2 = ...
        if parser.has_section('CameraGeometry'):
            dists, order, los = self._read_camera_geometry_from_cfg(parser)
            config.camera_distances = dists
            config.camera_distance_weight = CAMERA_DISTANCE_WEIGHT   # fixed, not user-set
            config.camera_cyclic_order = order
            config.camera_los = los
            if los:
                logger.info(f"Camera-glass line-of-sight check ENABLED: {len(los)} camera(s) "
                            f"(in-situ validity check, not a penalty)")
            if dists:
                logger.info(
                    f"Camera-geometry constraint ENABLED: {len(dists)} known "
                    f"inter-camera distance(s), weight={CAMERA_DISTANCE_WEIGHT:g} px/mm (fixed)"
                    + (f", cyclic order {order} (reflection check ON)" if order
                       else " (no cyclic order -> reflection check OFF)")
                )
            if parser.has_option('CameraGeometry', 'Weight'):
                logger.info("[CameraGeometry] 'Weight' is ignored — the penalty weight "
                            "is fixed in code and cannot be set from the config.")
        else:
            logger.info("No [CameraGeometry] section found — camera scale unconstrained.")
        if getattr(config, 'camera_glass_auto', False):
            _n_los = len(getattr(config, 'camera_los', None) or [])
            _n_d = len(getattr(config, 'camera_distances', None) or [])
            if _n_los < 3 or _n_d < 3:
                raise ValueError(
                    "Camera-Glass-Distance: auto needs the line-of-sight distances (LOS-i) of at "
                    "least 3 cameras and the inter-camera distances (Dist-i-j) in [CameraGeometry]; "
                    f"found {_n_los} LOS and {_n_d} distances")
            logger.info("Camera-Glass-Distance: auto - the camera-glass height of every seed frame is "
                        f"DERIVED from the {_n_los} line-of-sight distances")

        return config

    # -----------------------------------------------------------------------
    # _read_camera_geometry_from_cfg
    # -----------------------------------------------------------------------
    # Reads Dist-i-j keys (0-based camera indices) as pairwise distances in mm,
    # and an optional CyclicOrder (comma/space-separated 0-based indices). Any
    # 'Weight' key is ignored (the penalty weight is fixed in code). Missing/blank
    # section -> empty list / empty order.
    def _read_camera_geometry_from_cfg(self, parser) -> tuple:
        """Extract known camera distances, cyclic order and line-of-sight
        camera-glass distances (LOS-i, 0-based) from [CameraGeometry]."""
        dists = []
        order = []
        los = []
        for key, raw in parser.items('CameraGeometry'):
            k = key.strip().lower()
            if k.startswith('los-'):
                try:
                    i = int(k.split('-')[1]); d = float(raw)
                except (ValueError, TypeError, IndexError):
                    logger.warning(f"[CameraGeometry] ignoring malformed key '{key}'")
                    continue
                if d > 0:
                    los.append((i, d))
                continue
            if k == 'cyclicorder':
                try:
                    order = [int(x) for x in re.split(r'[,\s]+', raw.strip()) if x != '']
                except (ValueError, TypeError):
                    logger.warning(f"[CameraGeometry] ignoring malformed CyclicOrder '{raw}'")
                    order = []
                continue
            if not k.startswith('dist-'):
                continue      # 'weight' and any other key silently ignored
            try:
                _, si, sj = k.split('-')
                i, j = int(si), int(sj)
                d = float(raw)
            except (ValueError, TypeError):
                logger.warning(f"[CameraGeometry] ignoring malformed key '{key}'")
                continue
            if i != j and d > 0:
                dists.append((i, j, d))
        return dists, order, sorted(los)

    # -----------------------------------------------------------------------
    # _load_from_cfg — multimedia extension
    # -----------------------------------------------------------------------
    # The [Multimedia] section is optional.  If it is absent from the .cfg
    # file (e.g. an existing air-only experiment), all defaults (n1=1.0, etc.)
    # are kept and use_multimedia returns False → no behaviour change.
    #
    # Section format in the .cfg file:
    #
    #   [Multimedia]
    #   # Refractive index of air (camera side) — almost always 1.0
    #   N1 = 1.0
    #   # Refractive index of the glass wall — space-separated for multi-layer
    #   N2 = 1.5
    #   # Thickness of each glass layer in mm — must match number of N2 values
    #   D  = 5.0
    #   # Refractive index of water (object side)
    #   N3 = 1.333
    #   # Z-coordinate of the glass top surface in world coordinates (mm)
    #   Z-Glass-Top = 0.0
    #
    # To DISABLE multimedia (revert to standard pinhole):
    #   Delete the [Multimedia] section entirely, OR
    #   set N2 = 1.0 and N3 = 1.0
    def _read_multimedia_from_cfg(self, parser) -> dict:
        """Extract multimedia keys from an already-read ConfigParser."""
        def get_float(key, default):
            return float(parser.get('Multimedia', key, fallback=str(default)))

        def get_float_list(key, default):
            raw = parser.get('Multimedia', key, fallback='')
            if not raw:
                return default
            return [float(x) for x in raw.split()]

        def get_int(key, default):
            try:
                return int(float(parser.get('Multimedia', key, fallback=str(default))))
            except (ValueError, TypeError):
                return default

        return dict(
            n1          = get_float('N1', 1.0),
            n2          = get_float_list('N2', [1.0]),
            d           = get_float_list('D',  [0.0]),
            n3          = get_float('N3', 1.0),
            z_glass_top = get_float('Z-Glass-Top', 0.0),
            camera_glass_auto = parser.get('Multimedia', 'Camera-Glass-Distance', fallback='').strip().lower() == 'auto',
            camera_glass_distance_mm = (0.0 if parser.get('Multimedia', 'Camera-Glass-Distance', fallback='').strip().lower() == 'auto' else get_float('Camera-Glass-Distance', 0.0)),
            lut_n_r     = get_int('LUT-nR', 80),
            lut_n_zp    = get_int('LUT-nZp', 80),
            lut_zp_max  = get_float('LUT-Zp-Max', 300.0),
            lut_r_max   = get_float('LUT-R-Max', 0.0),
            ba_cam_params = self._parse_ba_params(
                parser.get('Multimedia', 'BA-Params', fallback='')),
            ba_mode = self._parse_ba_mode(parser.get('Multimedia', 'BA-Mode', fallback='')),
            seed_max_radius = max(0.0, get_float('Seed-Max-Radius', 0.0)),
            seed_subset_frac = min(1.0, max(0.3, get_float('Seed-Subset-Fraction', 1.0))),
            derefract_max_shift_px = max(0.0, get_float('Derefract-Max-Shift-Px', 0.0)),
            handover_max_reproj_px = max(0.1, get_float('Handover-Max-Reproj-Px', 1.0)),
            dist_weight_final = max(0.0, get_float('Dist-Weight-Final', CAMERA_DISTANCE_WEIGHT)),
            dist_misfit_max = max(0.1, get_float('Dist-Misfit-Max', 4.0)),
            intrinsics_mode = ('estimate' if parser.get('Multimedia', 'Intrinsics', fallback='rad').strip().lower() == 'estimate' else 'rad'),
            focal_start_px = max(0.0, get_float('Focal-Start-Px', 0.0)),
            pp_prior_sigma_px = max(0.1, get_float('PP-Prior-Sigma-Px', 8.0)),
            los_sigma_final_mm = max(0.0, get_float('LOS-Sigma-Final-Mm', 0.0)),
            ba_all_particles = parser.get('Multimedia', 'BA-All-Particles', fallback='0').strip().lower() in ('1', 'true', 'yes', 'on'),
        )

    _BA_PARAM_NAMES = ['X0', 'Y0', 'Z0', 'omega', 'phi', 'kappa', 'c', 'xh', 'yh',
                       'k1', 'k2', 'k3', 'p1', 'p2', 'a1', 'a2']

    _BA_MODES = ('alternate', 'joint', 'camera-then-joint')

    def _parse_ba_mode(self, raw: str) -> str:
        """BA-Mode: alternate | joint | camera-then-joint (default alternate)."""
        m = (raw or '').strip().lower()
        if not m:
            return 'alternate'
        if m not in self._BA_MODES:
            logger.warning(f"[Multimedia] BA-Mode '{raw}' unknown - using 'alternate' "
                           f"(allowed: {', '.join(self._BA_MODES)})")
            return 'alternate'
        return m

    def _parse_ba_params(self, raw: str) -> list:
        """'X0 Y0 Z0 omega phi kappa c' -> [0,1,2,3,4,5,6]. Empty -> the 6
        exterior parameters. Unknown names, and k1..a2 (not used by the
        projection under observation pre-correction), are rejected with a
        warning."""
        default = [0, 1, 2, 3, 4, 5]
        names = [t for t in re.split(r'[,\s]+', (raw or '').strip()) if t]
        if not names:
            return default
        lower = [n.lower() for n in self._BA_PARAM_NAMES]
        out = []
        for n in names:
            if n.lower() not in lower:
                logger.warning(f"[Multimedia] BA-Params: unknown parameter '{n}' ignored")
                continue
            idx = lower.index(n.lower())
            if idx > 8:
                logger.warning(f"[Multimedia] BA-Params: '{n}' ignored - distortion/A-D "
                               f"are corrected on the observations, not optimised in the BA")
                continue
            if idx not in out:
                out.append(idx)
        return sorted(out) if out else default

    def _load_from_json(self, filepath: Path) -> CalibrationConfig:
        """Load configuration from JSON file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        return CalibrationConfig(**data)

    def save_config(self, config: CalibrationConfig, filepath: Union[str, Path],
                    format: str = 'auto') -> None:
        """
        Save configuration to file.

        Examples:
            system.save_config(config, 'output.cfg')
            system.save_config(config, 'output.json')
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        if format == 'auto':
            if filepath.suffix == '.json':
                format = 'json'
            else:
                format = 'cfg'

        if format == 'cfg':
            self._save_as_cfg(config, filepath)
        elif format == 'json':
            self._save_as_json(config, filepath)
        else:
            raise ValueError(f"Unsupported format: {format}")

        logger.info(f"Saved config to {filepath} ({format} format)")

    def _save_as_cfg(self, config: CalibrationConfig, filepath: Path) -> None:
        """Save configuration as .cfg file (INI format)."""
        cfg = configparser.ConfigParser()

        def list_str(lst):
            return '   '.join(map(str, lst))

        if config.data_path or config.img_path:
            cfg['PATHS'] = {}
            if config.data_path:
                cfg['PATHS']['Data-Path'] = config.data_path
            if config.img_path:
                cfg['PATHS']['Image-Path'] = config.img_path

        cfg['Files'] = {}
        if config.basename:
            cfg['Files']['Basename'] = config.basename
        if config.img_extension:
            cfg['Files']['Image-Extension'] = config.img_extension
        if config.img_names:
            cfg['Files']['Image-Names'] = config.img_names

        cfg['Images'] = {'Subpix': str(config.subpix)}
        if config.led_size != 7:
            cfg['Images']['LED-Size'] = str(config.led_size)
        if config.led_color != 'green':
            cfg['Images']['LED-Color'] = config.led_color
        if config.led_threshold != 30:
            cfg['Images']['LED-Threshold'] = str(config.led_threshold)

        cfg['Calibration'] = {
            'Num-Cameras': str(config.num_cameras),
            'Num-Projectors': str(config.num_projectors),
            'Nonlinear-Parameters': list_str(config.nonlinear_parameters),
            'Nonlinear-Update': list_str(config.nonlinear_update),
            'Do-Global-Iterations': '1' if config.do_global_iterations else '0',
            'Global-Iteration-Threshold': str(config.global_iteration_threshold),
            'Global-Iteration-Max': str(config.global_iteration_max),
            'Num-Cameras-Fill': str(config.num_cameras_fill),
            'Do-Bundle-Adjustment': '1' if config.do_bundle_adjustment else '0',
            'Undo-Radial': '1' if config.undo_radial else '0',
            'Min-Points-Value': str(config.min_points_value),
            'N-Tuples': str(config.n_tuples),
            'Square-Pixels': '1' if config.square_pixels else '0',
            'Use-Nth-Frame': str(config.use_nth_frame)
        }

        if config.initial_tolerance != 10.0:
            cfg['Calibration']['Initial-Tolerance'] = str(config.initial_tolerance)
        if config.align_existing:
            cfg['Calibration']['Align-Existing'] = '1'

        # -------------------------------------------------------------------
        # [Multimedia] section — written only when multimedia is active.
        # -------------------------------------------------------------------
        # Why only when active: existing .cfg files loaded by older versions
        # of this code must remain unchanged.  An absent [Multimedia] section
        # is equivalent to n1=1, n2=[1], d=[0], n3=1 (air only).
        #
        # Multi-layer glass (e.g. two panes) is stored as space-separated
        # values on N2 and D lines:
        #   N2 = 1.5 1.5
        #   D  = 4.0 4.0
        if config.use_multimedia:
            cfg['Multimedia'] = {
                # N1 — camera-side medium (air).  Almost always 1.0.
                'N1': str(config.n1),
                # N2 — glass layer(s).  Space-separated for multi-layer stacks.
                'N2': '   '.join(str(v) for v in config.n2),
                # D  — glass thickness in mm per layer, same length as N2.
                'D':  '   '.join(str(v) for v in config.d),
                # N3 — object-side medium (water ≈ 1.333 at 20°C).
                'N3': str(config.n3),
                # Z-Glass-Top — Z coordinate (mm) of the air/glass interface
                # in the world coordinate system.
				'Z-Glass-Top': str(config.z_glass_top),
                'Camera-Glass-Distance': ('auto' if getattr(config, 'camera_glass_auto', False)
                                          else str(config.camera_glass_distance_mm)),
            }

        with open(filepath, 'w') as f:
            cfg.write(f)

    def _save_as_json(self, config: CalibrationConfig, filepath: Path) -> None:
        """Save configuration as JSON file.

        Excludes class-level attributes (principal_point), properties
        (use_multimedia), and methods (to_multimedia_params) — only
        dataclass instance fields are serialised.
        """
        import dataclasses
        data = {}
        for f in dataclasses.fields(config):
            val = getattr(config, f.name)
            # Convert tuples/numpy arrays to plain lists for JSON
            if isinstance(val, (tuple, np.ndarray)):
                val = list(val)
            data[f.name] = val
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

    def load_data(self, config: Optional[CalibrationConfig] = None) -> CalibrationData:
        """Load calibration data from files."""
        if config is None:
            if self.current_config is None:
                raise ValueError("No configuration loaded. Call load_config() first.")
            config = self.current_config

        data_path = Path(config.data_path)

        # Load main data files
        Ws = self._load_data_file(data_path / 'points.dat')
        id_matrix = self._load_data_file(data_path / 'IdMat.dat')

        # Validate config vs actual data dimensions
        actual_cams = Ws.shape[0] // 3
        config_cams = len(config.cameras_2_use)
        if actual_cams != config_cams:
            raise ValueError(
                f"\n{'='*60}\n"
                f"CONFIG / DATA MISMATCH\n"
                f"{'='*60}\n"
                f"  Config says:    {config_cams} cameras "
                f"(Num-Cameras: {config.num_cameras}, "
                f"Cameras_to_use: {list(config.cameras_2_use)})\n"
                f"  points.dat has: {actual_cams} cameras "
                f"({Ws.shape[0]} rows ÷ 3 = {actual_cams}, "
                f"{Ws.shape[1]} frames)\n"
                f"\n"
                f"  Please fix your config file:\n"
                f"    [Calibration]\n"
                f"    Num-Cameras: {actual_cams}\n"
                f"{'='*60}"
            )

        # Load resolutions (optional)
        res_file = data_path / 'Res.dat'
        if res_file.exists():
            resolutions = self._load_data_file(res_file)
            _res_src = 'Res.dat'
        else:
            resolutions = np.tile(config.resolution, (config.num_cameras, 1))
            _res_src = "NO Res.dat - config [Images] Resolution or its default 640x480"
        _r0 = np.atleast_2d(resolutions)[0]
        logger.info(f"Image resolution: {int(_r0[0])}x{int(_r0[1])} (from {_res_src})")

        # Apply frame resampling if needed
        if config.use_nth_frame > 1:
            Ws = Ws[:, ::config.use_nth_frame]
            id_matrix = id_matrix[:, ::config.use_nth_frame]

        # Load camera parameters from .rad files
        logger.info("Looking for camera parameter files (.rad)...")
        camera_parameters = self._load_camera_parameters(data_path, config.basename, config.num_cameras)
        if camera_parameters:
            flag = True
        else:
            flag = False
        data = CalibrationData(
            Ws=Ws,
            id_matrix=id_matrix,
            resolutions=resolutions,
            projection_matrices=None,
            camera_parameters=camera_parameters,
            INT_CAM_PARAMS=flag
        )

        self.current_data = data
        logger.info(f"Loaded data: {Ws.shape[1]} frames, {config.num_cameras} cameras")

        if camera_parameters:
            logger.info(f"  ✅ Loaded {len(camera_parameters)} camera parameter files (.rad)")

        return data

    def _load_data_file(self, filepath: Path) -> np.ndarray:
        """Load data file with fallback strategies."""
        if not filepath.exists():
            raise FileNotFoundError(f"Data file not found: {filepath}")

        strategies = [
            lambda: np.loadtxt(filepath),
            lambda: np.loadtxt(filepath, delimiter=','),
            lambda: np.loadtxt(filepath, delimiter='\t'),]

        for strategy in strategies:
            try:
                return strategy()
            except:
                continue

        raise IOError(f"Could not load file: {filepath}")

    def _load_rad_file(self, filepath: Path) -> Dict[str, float]:
        """
        Load .rad file (camera radial distortion file).

        .rad files contain camera parameters in key=value format:
            K11 = 422.202325
            K12 = 0.000000
            ...
            kc1 = -0.280971
            ...

        Args:
            filepath: Path to .rad file

        Returns:
            dict: Dictionary with parameter names and values
        """
        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")

        params = {}

        with open(filepath, 'r') as f:
            for line in f:
                line = line.strip()
                # Skip empty lines and comments
                if not line or line.startswith('#') or line.startswith('%'):
                    continue

                # Parse key = value
                if '=' in line:
                    key, value = line.split('=', 1)
                    key = key.strip()
                    value = value.strip()

                    try:
                        params[key] = float(value)
                    except ValueError:
                        logger.warning(f"Could not parse value for {key}: {value}")

        if not params:
            raise ValueError("No parameters found in .rad file")

        return params

    def _load_camera_parameters(self, data_path: Path, basename: str, num_cameras: int) -> Optional[List[Dict]]:
        """
        Load camera intrinsic parameters and distortion from .rad files.

        Files are named: {basename}{camera_number}.rad
        Example: basename1.rad, basename2.rad, basename3.rad, basename4.rad

        .rad files contain parameters in key=value format:
            K11, K12, K13, K21, K22, K23, K31, K32, K33 (intrinsic matrix)
            kc1, kc2, kc3, kc4, kc5 (distortion coefficients)
        """
        camera_params = []

        for cam_id in range(1, num_cameras + 1):
            rad_file = data_path / f'{basename}{cam_id}.rad'

            if not rad_file.exists():
                logger.warning(f"Camera parameter file not found: {rad_file}")
                return None

            try:
                # Load the .rad file as dictionary
                params_dict = self._load_rad_file(rad_file)

                # Build intrinsic matrix from K11, K12, K13, K21, K22, K23, K31, K32, K33
                intrinsic_matrix = np.array([
                    [params_dict.get('K11', 0.0), params_dict.get('K12', 0.0), params_dict.get('K13', 0.0)],
                    [params_dict.get('K21', 0.0), params_dict.get('K22', 0.0), params_dict.get('K23', 0.0)],
                    [params_dict.get('K31', 0.0), params_dict.get('K32', 0.0), params_dict.get('K33', 1.0)]
                ])

                # Extract distortion coefficients (kc1, kc2, kc3, kc4, kc5, ...)
                distortion = []
                for i in range(1, 10):  # Try up to kc9
                    key = f'kc{i}'
                    if key in params_dict:
                        distortion.append(params_dict[key])
                    else:
                        break  # Stop when we don't find the next coefficient

                # Ensure we have exactly 6 distortion parameters (pad with zeros if needed)
                while len(distortion) < 6:
                    distortion.append(0.0)

                distortion = np.array(distortion[:6])  # Take only first 6

                params = {
                    'camera_id': cam_id,
                    'intrinsic_matrix': intrinsic_matrix,
                    'distortion': distortion,
                    'file': str(rad_file),
                    'raw_params': params_dict  # Keep original params for reference
                }

                camera_params.append(params)
                logger.info(f"  ✅ Loaded camera {cam_id} parameters from {rad_file.name}")
                logger.info(
                    f"     fx={intrinsic_matrix[0, 0]:.2f}, fy={intrinsic_matrix[1, 1]:.2f}, cx={intrinsic_matrix[0, 2]:.2f}, cy={intrinsic_matrix[1, 2]:.2f}")
                if len(distortion) > 0:
                    logger.info(f"     Distortion: {len(distortion)} coefficients")

            except Exception as e:
                logger.warning(f"Could not load {rad_file}: {e}")

        return camera_params if camera_params else None

    def save_data(self, data: Optional[CalibrationData] = None,
                  output_path: Union[str, Path] = None,
                  delimiter: str = ' ') -> None:
        """Save calibration data to files."""
        if data is None:
            if self.current_data is None:
                raise ValueError("No data loaded.")
            data = self.current_data

        if output_path is None:
            if self.current_config is None:
                raise ValueError("No output path specified.")
            output_path = self.current_config.data_path

        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)

        np.savetxt(output_path / 'points.dat', data.Ws, delimiter=delimiter, fmt='%.6f')
        np.savetxt(output_path / 'IdMat.dat', data.id_matrix, delimiter=delimiter, fmt='%d')
        np.savetxt(output_path / 'Res.dat', data.resolutions, delimiter=delimiter, fmt='%d')

        if data.projection_matrices:
            for i, pmat in enumerate(data.projection_matrices):
                np.savetxt(output_path / f'camera{i + 1}.Pmat.cal', pmat, delimiter=delimiter, fmt='%.6f')

        logger.info(f"Saved data to {output_path}")


    def load_config_from_file(self, paths_file: str) -> CalibrationConfig:
        """
        Load configuration from a simple text file containing paths.

        The text file should contain:
            data_folder=/path/to/data/
            config_file=/path/to/experiment.cfg

        Args:
            paths_file: Path to the text file with folder/file paths

        Returns:
            CalibrationConfig: Loaded configuration
        """
        from pathlib import Path

        paths = {}
        with open(paths_file, 'r') as f:
            for line in f:
                line = line.strip()
                # Skip comments and empty lines
                if line.startswith('#') or not line:
                    continue
                # Parse key=value
                if '=' in line:
                    key, value = line.split('=', 1)
                    paths[key.strip()] = value.strip()

        # Get the paths
        data_folder = paths.get('data_folder', '')
        config_file = paths.get('config_file', '')

        if not config_file:
            raise ValueError("config_file not specified in paths file")

        # Load the .cfg file
        config = self.load_config(config_file)

        # Override data_path if specified
        if data_folder:
            config.data_path = data_folder

        # Validate that required files exist
        data_path = Path(config.data_path)
        required_files = ['points.dat', 'IdMat.dat']

        missing = []
        for filename in required_files:
            if not (data_path / filename).exists():
                missing.append(filename)

        if missing:
            logger.warning(f"Missing files: {', '.join(missing)}")

        logger.info(f"Configuration loaded from: {config_file}")
        logger.info(f"Data folder: {config.data_path}")

        return config

    def save_results(self, config: CalibrationConfig,
                     data: CalibrationData,
                     output_folder: str = 'Results') -> None:
        """Save all calibration results to a Results folder."""
        from pathlib import Path

        # Create Results folder in the same directory as the data folder
        if config.data_path:
            base_dir = Path(config.data_path).parent
            output_path = base_dir / output_folder
        else:
            output_path = Path(output_folder)

        output_path.mkdir(parents=True, exist_ok=True)

        logger.info(f"Saving results to: {output_path}")

        # 1. Save configuration
        self.save_config(config, output_path / 'config.cfg', format='cfg')
        logger.info("  ✅ Saved config.cfg")

        self.save_config(config, output_path / 'config.json', format='json')
        logger.info("  ✅ Saved config.json")

        # 2. Save calibration data
        self.save_data(data, output_path, delimiter=' ')
        logger.info("  ✅ Saved calibration data files")

        # 3. Save camera parameters (.rad files) if available
        if data.camera_parameters:
            for cam_param in data.camera_parameters:
                cam_id = cam_param['camera_id']
                rad_file = output_path / f"{config.basename}{cam_id}.rad"

                # Write in key=value format
                with open(rad_file, 'w') as f:
                    K = cam_param['intrinsic_matrix']

                    # Write intrinsic matrix
                    f.write(f"K11 = {K[0, 0]:.6f}\n")
                    f.write(f"K12 = {K[0, 1]:.6f}\n")
                    f.write(f"K13 = {K[0, 2]:.6f}\n")
                    f.write(f"K21 = {K[1, 0]:.6f}\n")
                    f.write(f"K22 = {K[1, 1]:.6f}\n")
                    f.write(f"K23 = {K[1, 2]:.6f}\n")
                    f.write(f"K31 = {K[2, 0]:.6f}\n")
                    f.write(f"K32 = {K[2, 1]:.6f}\n")
                    f.write(f"K33 = {K[2, 2]:.6f}\n")

                    # Write distortion coefficients
                    distortion = cam_param['distortion']
                    for i, kc in enumerate(distortion, start=1):
                        f.write(f"kc{i} = {kc:.6f}\n")

                logger.info(f"  ✅ Saved {rad_file.name}")

        # 4. Create a summary text file
        summary_file = output_path / 'summary.txt'
        with open(summary_file, 'w') as f:
            f.write("Multi-Camera Calibration Results\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Experiment: {config.basename}\n")
            f.write(f"Number of Cameras: {config.num_cameras}\n")
            f.write(f"Number of Frames: {data.Ws.shape[1]}\n")
            f.write(f"Subpixel Accuracy: {config.subpix}\n\n")

            # ---------------------------------------------------------------
            # Multimedia geometry summary in the text report
            # ---------------------------------------------------------------
            # This block is intentionally verbose so that the summary.txt
            # file is a self-contained record of the physical setup —
            # important for reproducing results months or years later.
            f.write("Multimedia Geometry:\n")
            if config.use_multimedia:
                f.write("  Status  : ENABLED (through-glass / underwater)\n")
                f.write(f"  n1 (air,   camera side) : {config.n1}\n")
                f.write(f"  n2 (glass, wall layers) : {config.n2}\n")
                f.write(f"  d  (glass thickness mm) : {config.d}\n")
                f.write(f"  n3 (fluid, object side) : {config.n3}\n")
                f.write(f"  z_glass_top (mm)        : {config.z_glass_top}\n\n")
            else:
                f.write("  Status  : DISABLED (standard pinhole, air only)\n\n")

            f.write("Calibration Settings:\n")
            f.write(f"  - Bundle Adjustment: {config.do_bundle_adjustment}\n")
            f.write(f"  - Global Iterations: {config.do_global_iterations}\n")
            f.write(f"  - N-tuples: {config.n_tuples}\n")
            f.write(f"  - Min Points: {config.min_points_value}\n")
            f.write(f"  - Use Nth Frame: {config.use_nth_frame}\n\n")

            f.write("Files Generated:\n")
            f.write("  - config.cfg, config.json\n")
            f.write("  - points.dat, IdMat.dat, Res.dat\n")

            if data.camera_parameters:
                f.write(f"  - {len(data.camera_parameters)} camera parameter files (.rad)\n")
                for cam_param in data.camera_parameters:
                    f.write(f"    • {config.basename}{cam_param['camera_id']}.rad\n")

            if data.projection_matrices:
                f.write(f"  - {len(data.projection_matrices)} camera projection matrices\n")

        logger.info("  ✅ Saved summary.txt")
        logger.info(f"\n🎉 All results saved to: {output_path.absolute()}")


# Quick API functions
def load_config(filepath: Union[str, Path]) -> CalibrationConfig:
    """Quick function to load configuration."""
    system = CalibrationSystem()
    return system.load_config(filepath)


def save_config(config: CalibrationConfig, filepath: Union[str, Path], format: str = 'auto') -> None:
    """Quick function to save configuration."""
    system = CalibrationSystem()
    system.save_config(config, filepath, format)


def load_data(config: CalibrationConfig) -> CalibrationData:
    """Quick function to load data."""
    system = CalibrationSystem()
    return system.load_data(config)

def _intrinsics_from_Pe(Pe_k, imsize=None):
    """
    Extract focal length and principal point from a 3×4 Euclidean camera matrix.

    Pe_k = K @ [R | t]  ⟹  Pe_k[:, :3] = K @ R_cam

    Uses CoreFunctions.RQ_decomposition.rq() for the RQ decomposition.

    NORMALISATION NOTE
    ------------------
    bundle_PX_proj applies normP() to P and the VGG conditioner H_k before
    the LM solve.  The unconditioned Pe returned after bundle adjustment still
    carries the normP scale factor.  As a result, the RQ-decomposed focal
    length c_rq is in normalised units, NOT in pixels.

    To recover the true pixel focal length we multiply by the conditioner
    scale f = (W+H)/2, because the conditioner divides pixel coordinates
    by this factor before the LM solve, and unconditiong multiplies them
    back — but normP introduces an additional scale that makes c_rq small.

    Empirically (confirmed from log):  c_true_pixels ≈ c_rq × (W+H)/2

    If imsize (W, H) is not provided the conditioner scale cannot be applied
    and the raw c_rq is returned — this will be ~1-2 px, too small for
    multimedia operations, which will be safely caught by _validate_geometry.

    Returns (c, xh, yh) in pixel units, or (None, None, None) on failure.
    """
    from CoreFunctions.RQ_decomposition import rq
    try:
        K_raw, _ = rq(Pe_k[:, :3].copy())
        s = K_raw[2, 2]
        if abs(s) < 1e-10:
            raise ValueError("K[2,2] ~ 0")
        K_norm = K_raw / s          # normalise so K[2,2] = 1
        c_rq   = abs(float(K_norm[0, 0]))   # abs handles sign flip from rq()
        xh_rq  = float(K_norm[0, 2])
        yh_rq  = float(K_norm[1, 2])

        # Apply conditioner scale to recover true pixel focal length
        if imsize is not None and len(imsize) >= 2:
            W, H   = float(imsize[0]), float(imsize[1])
            if c_rq > 20.0:
                # Pe already maps to principal-point-CENTRED PIXEL coordinates (the
                # pipeline's camera matrices after the metric upgrade): the focal length
                # is in pixels as it stands. Run evidence: multiplying by (W+H)/2 here
                # gave focal lengths of 250,000-620,000 px.
                c  = c_rq
                xh = xh_rq + W / 2.0
                yh = yh_rq + H / 2.0
            else:
                # conditioned units (c of order 1): undo the VGG conditioner scale
                f_cond = (W + H) / 2.0
                c  = c_rq  * f_cond
                xh = xh_rq * f_cond + W / 2.0   # shift principal point back
                yh = yh_rq * f_cond + H / 2.0
        else:
            c, xh, yh = c_rq, xh_rq, yh_rq

        if c < 1.0 or c > 1e7:
            raise ValueError(f"Implausible focal length: {c}")
        return c, xh, yh
    except Exception:
        return None, None, None   # caller will use image-size fallback


def build_camera_params(Ce, Re, config: CalibrationConfig, mm, CAMS, Pe=None):
    """
    Build one CameraParams per camera from the Euclidean reconstruction output.

    Intrinsic parameters (focal length c, principal point xh/yh) are sourced
    in the following priority order:

        1. .rad calibration files  (config._camera_parameters)
           → use when physical calibration is available.

        2. RQ decomposition of Pe  (the Euclidean camera matrix from euclidize)
           → use in the normal self-calibration scenario where no .rad files
             exist.  euclidize() already encodes the metric intrinsics inside Pe;
             we just need to decompose it.

        3. Image-size fallback
           → last resort when both above fail.  Uses half the image width as a
             rough focal-length estimate.  Adequate to get the right order of
             magnitude for the multimedia geometry.

    Parameters
    ----------
    Ce : np.ndarray (4, CAMS)
        Homogeneous camera centres from euclidize().
    Re : np.ndarray (3*CAMS, 3)
        Stacked rotation matrices from euclidize().
    config : CalibrationConfig
    mm : MultimediaParams
    CAMS : int
    Pe : np.ndarray (3*CAMS, 4), optional
        Euclidean projection matrices from euclidize().
        Provide this so that intrinsics can be extracted when .rad files are
        absent — which is the typical self-calibration scenario.
    """
    from Multimedia.Multimedia_params import CameraParams
    cameras = []

    for i in range(CAMS):
        position = Ce[:3, i]   # metric world mm (drop homogeneous row)
        R_i = Re[3*i:3*i+3, :]
        try:
            from scipy.spatial.transform import Rotation
            angles = Rotation.from_matrix(R_i).as_euler('xyz')
        except ImportError:
            phi   = np.arcsin(-R_i[2, 0])
            omega = np.arctan2(R_i[2, 1], R_i[2, 2])
            kappa = np.arctan2(R_i[1, 0], R_i[0, 0])
            angles = np.array([omega, phi, kappa])

        # ── Source 1: .rad calibration file ──────────────────────────────────
        have_rad = (hasattr(config, 'camera_parameters') and
                    config.camera_parameters and
                    i < len(config.camera_parameters))
        dist = np.zeros(6)
        if have_rad:
            K_rad = config.camera_parameters[i]['intrinsic_matrix']
            dist  = config.camera_parameters[i]['distortion']
            c     = float(K_rad[0, 0])
            xh    = float(K_rad[0, 2])
            yh    = float(K_rad[1, 2])
            source = 'rad'

        # ── Source 2: RQ decomposition of Pe ─────────────────────────────────
        elif Pe is not None:
            Pe_k = Pe[3*i:3*i+3, :]
            # Pass image size so _intrinsics_from_Pe can apply conditioner scale
            res_i = (config.resolutions[i] if hasattr(config, 'resolutions')
                     and config.resolutions is not None
                     else getattr(config, 'resolution', None))
            imsize_i = res_i[:2] if res_i is not None else None
            c_rq, xh_rq, yh_rq = _intrinsics_from_Pe(Pe_k, imsize=imsize_i)
            if c_rq is not None:
                c, xh, yh = c_rq, xh_rq, yh_rq
                source = 'Pe_rq'
                if getattr(config, 'intrinsics_mode', 'rad') == 'estimate' and imsize_i is not None:
                    # estimate mode: the seed gives the focal length; the principal point
                    # starts at the image centre (the final stages refine both)
                    xh, yh = float(imsize_i[0]) / 2.0, float(imsize_i[1]) / 2.0
                    source = 'Pe_rq focal + image centre'
            else:
                # RQ failed — fall through to image-size fallback
                have_rad = False   # trigger fallback below
                c = xh = yh = None
                source = 'fallback'
        else:
            c = xh = yh = None
            source = 'fallback'

        # estimate mode with an explicit focal start value (e.g. from the lens datasheet)
        if (getattr(config, 'intrinsics_mode', 'rad') == 'estimate'
                and float(getattr(config, 'focal_start_px', 0.0) or 0.0) > 0 and not have_rad):
            _r = (config.resolutions[i] if getattr(config, 'resolutions', None) is not None
                  else getattr(config, 'resolution', (640, 480)))
            c = float(config.focal_start_px)
            xh, yh = float(_r[0]) / 2.0, float(_r[1]) / 2.0
            source = 'Focal-Start-Px + image centre'

        # ── Source 3: image-size fallback ─────────────────────────────────────
        if source == 'fallback' or c is None:
            res = getattr(config, 'resolution', None)
            if res is not None and len(res) >= 2:
                W, H = int(res[0]), int(res[1])
            elif hasattr(config, 'resolutions') and config.resolutions is not None:
                W, H = int(config.resolutions[0, 0]), int(config.resolutions[0, 1])
            else:
                W, H = 640, 480
            # Half the image width is a reasonable order-of-magnitude estimate
            c  = float(W) / 2.0
            xh = float(W) / 2.0
            yh = float(H) / 2.0
            source = 'fallback'
            import warnings
            warnings.warn(
                f"Camera {i}: no .rad file and RQ decomp of Pe failed — "
                f"using image-size fallback (c={c:.1f}, xh={xh:.1f}, yh={yh:.1f}). "
                f"Multimedia BA results may be approximate.",
                RuntimeWarning,
            )

        cam_obj = CameraParams(
            cam_id   = config.cameras_2_use[i] if config.cameras_2_use else i+1,
            position = position,
            angles   = angles,
            c        = c,
            xh       = xh,
            yh       = yh,
            # Lens distortion and A/D are corrected ONCE on the observations
            # (GoCal _correct_observations, Caltech kc convention). The multimedia
            # model is a pinhole on corrected coordinates, so these fields must be 0:
            # filling them (previously with Caltech kc in Maas order) was never
            # applied and only mislabelled the outputs.
            k1=0.0, k2=0.0, k3=0.0, p1=0.0, p2=0.0,
            a1=0.0, a2=0.0,
            mm=mm,
        )
        cameras.append(cam_obj)
        logger.debug(f"Camera {i}: intrinsics from [{source}]  "
                     f"c={c:.2f}  xh={xh:.2f}  yh={yh:.2f}")
    return cameras