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
import configparser
import numpy as np
from pathlib import Path
from typing import List, Optional, Union, Tuple, Dict
from dataclasses import dataclass, field
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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
    align_existing: bool = False
    principal_point = np.zeros((num_cameras, 2))
    cameras_2_use: list[int] = field(default_factory=lambda: [])


    def __post_init__(self):
        """Normalize paths after initialization."""
        if self.data_path and not self.data_path.endswith(os.sep):
            self.data_path += os.sep
        if self.img_path and not self.img_path.endswith(os.sep):
            self.img_path += os.sep


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
                return default or []
            return [dtype(x) for x in value.split()]

        return CalibrationConfig(
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
            align_existing=get_bool('Calibration', 'Align-Existing', False),
            cameras_2_use=tuple(get_list('Calibration', 'Cameras_to_use', int, []))
        )

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

        with open(filepath, 'w') as f:
            cfg.write(f)

    def _save_as_json(self, config: CalibrationConfig, filepath: Path) -> None:
        """Save configuration as JSON file."""
        data = {k: list(v) if isinstance(v, tuple) else v
                for k, v in config.__dict__.items()}
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

        # Load resolutions (optional)
        res_file = data_path / 'Res.dat'
        if res_file.exists():
            resolutions = self._load_data_file(res_file)
        else:
            resolutions = np.tile(config.resolution, (config.num_cameras, 1))

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

    def create_default_config(self, basename: str = "experiment",
                              num_cameras: int = 4,
                              **kwargs) -> CalibrationConfig:
        """Create a new configuration with default values."""
        return CalibrationConfig(basename=basename, num_cameras=num_cameras, **kwargs)

    def print_config_summary(self, config: Optional[CalibrationConfig] = None) -> None:
        """Print a summary of the configuration."""
        if config is None:
            config = self.current_config

        if config is None:
            print("No configuration loaded")
            return

        print("=" * 60)
        print("CALIBRATION CONFIGURATION SUMMARY")
        print("=" * 60)
        print(f"Experiment: {config.basename}")
        print(f"Data path: {config.data_path}")
        print(f"\nCameras: {config.num_cameras}")
        print(f"Subpixel: {config.subpix}")
        print(f"\nCalibration:")
        print(f"  Bundle adjustment: {config.do_bundle_adjustment}")
        print(f"  Global iterations: {config.do_global_iterations} (max={config.global_iteration_max})")
        print(f"  N-tuples: {config.n_tuples}")
        print(f"  Min points: {config.min_points_value}")
        print(f"  Use every Nth frame: {config.use_nth_frame}")
        print("=" * 60)

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
