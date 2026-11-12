"""
Example demonstrating how to use the align_existing_camera_centers function.
This example shows typical usage with real camera calibration data.
"""

import numpy as np
import os
from align_existing_camera_centers import align_existing_camera_centers
from CommonCfgAndIO.CalibrationSystem import CalibrationConfig


def main():
    """
    Complete example showing how to use align_existing_camera_centers function
    with realistic camera calibration data.
    """

    print("=== Camera Alignment Example ===")

    # 1. Setup configuration using CalibrationConfig dataclass
    config = CalibrationConfig(
        data_path='./calibration_data/',
        cameras_2_use=[0, 1, 2, 3]
    )

    # Ensure data directory exists
    os.makedirs(config.data_path, exist_ok=True)

    # 2. Prepare input data
    num_cameras = 4
    num_3d_points = 150

    # Example: Camera centers from current calibration (3 x N_cameras)
    current_camera_centers = np.array([
        [100, 250, -50, 180],
        [200, -100, 300, -150],
        [1800, 1750, 1900, 1820]
    ])

    print(f"Current camera centers shape: {current_camera_centers.shape}")
    print(f"Current camera positions:\n{current_camera_centers}")

    # Example: Camera rotation matrices (N_cameras x 3 x 3)
    camera_rotations = np.zeros((num_cameras, 3, 3))
    for i in range(num_cameras):
        angle = i * np.pi / 8
        camera_rotations[i, :, :] = np.array([
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle), np.cos(angle), 0],
            [0, 0, 1]
        ])

    print(f"Camera rotations shape: {camera_rotations.shape}")

    # Example: 3D points from structure-from-motion or stereo reconstruction
    points_3d = np.random.randn(3, num_3d_points) * 300 + np.array([[0], [0], [1800]])

    print(f"3D points shape: {points_3d.shape}")
    print(f"3D points range - X: [{points_3d[0, :].min():.1f}, {points_3d[0, :].max():.1f}]")
    print(f"3D points range - Y: [{points_3d[1, :].min():.1f}, {points_3d[1, :].max():.1f}]")
    print(f"3D points range - Z: [{points_3d[2, :].min():.1f}, {points_3d[2, :].max():.1f}]")

    # Example: Camera projection matrices (list of 3x4 matrices)
    projection_matrices = []
    for i in range(num_cameras):
        K = np.array([[800, 0, 320],
                      [0, 800, 240],
                      [0, 0, 1]])

        R = camera_rotations[i, :, :]
        t = -R @ current_camera_centers[:, i]

        P = K @ np.hstack([R, t.reshape(-1, 1)])
        projection_matrices.append(P)

        print(f"Camera {i} projection matrix shape: {P.shape}")

    # Package input data
    input_data = {
        'Cst': current_camera_centers.T,
        'Rot': camera_rotations,
        'Pe': projection_matrices,
        'Xe': points_3d
    }

    print(f"\nInput data prepared:")
    print(f"- Camera centers: {input_data['Cst'].shape}")
    print(f"- Camera rotations: {input_data['Rot'].shape}")
    print(f"- Projection matrices: {len(input_data['Pe'])} matrices of shape {input_data['Pe'][0].shape}")
    print(f"- 3D points: {input_data['Xe'].shape}")

    # 3. Create reference camera centers file
    reference_cameras = np.array([
        [0, 200, -100, 150],
        [150, -50, 250, -200],
        [2000, 2000, 2000, 2000]
    ])

    ref_file_path = f"{config.data_path}original_cam_centers.dat"
    np.savetxt(ref_file_path, reference_cameras.T, fmt='%.6f')
    print(f"\nCreated reference camera file: {ref_file_path}")
    print(f"Reference camera positions:\n{reference_cameras}")

    # 4. Run the alignment
    print("\n" + "=" * 50)
    print("=== Running Camera Alignment ===")
    print("=" * 50)

    try:
        alignment_result = align_existing_camera_centers(input_data, config)

        # 5. Display results
        print("\n" + "=" * 40)
        print("=== Alignment Results ===")
        print("=" * 40)

        print(f"\nSimilarity transformation applied:")
        print(f"  Scale factor: {alignment_result['simT']['s']:.6f}")
        print(f"  Translation vector: [{alignment_result['simT']['t'][0]:.3f}, "
              f"{alignment_result['simT']['t'][1]:.3f}, {alignment_result['simT']['t'][2]:.3f}]")
        print(f"  Rotation matrix determinant: {np.linalg.det(alignment_result['simT']['R']):.6f}")

        print(f"\nBefore and after comparison:")
        print(f"Original camera centers:")
        for i, pos in enumerate(current_camera_centers.T):
            print(f"  Camera {i}: [{pos[0]:7.1f}, {pos[1]:7.1f}, {pos[2]:7.1f}]")

        print(f"\nAligned camera centers:")
        for i, pos in enumerate(alignment_result['Cst'].T):
            print(f"  Camera {i}: [{pos[0]:7.1f}, {pos[1]:7.1f}, {pos[2]:7.1f}]")

        print(f"\nReference camera centers:")
        for i, pos in enumerate(reference_cameras.T):
            print(f"  Camera {i}: [{pos[0]:7.1f}, {pos[1]:7.1f}, {pos[2]:7.1f}]")

        # Calculate alignment error
        alignment_error = np.linalg.norm(alignment_result['Cst'] - reference_cameras, axis=0)
        print(f"\nAlignment quality:")
        for i, error in enumerate(alignment_error):
            print(f"  Camera {i} error: {error:.3f} units")
        print(f"  Mean alignment error: {np.mean(alignment_error):.3f} units")
        print(f"  Max alignment error: {np.max(alignment_error):.3f} units")
        print(f"  RMS alignment error: {np.sqrt(np.mean(alignment_error ** 2)):.3f} units")

        # Check data consistency
        print(f"\nData consistency checks:")
        print(f"  Number of 3D points: {alignment_result['X'].shape[1]}")
        print(f"  Number of cameras: {alignment_result['Cst'].shape[1]}")
        print(f"  Aligned 3D points range:")
        print(f"    X: [{alignment_result['X'][0, :].min():.1f}, {alignment_result['X'][0, :].max():.1f}]")
        print(f"    Y: [{alignment_result['X'][1, :].min():.1f}, {alignment_result['X'][1, :].max():.1f}]")
        print(f"    Z: [{alignment_result['X'][2, :].min():.1f}, {alignment_result['X'][2, :].max():.1f}]")

        print(f"\nOutput files:")
        print(f"  Visualization plots saved to: {config.data_path}")
        print(f"  - topview.png: Top-down view of aligned scene")
        print(f"  - sideview.png: Side view of aligned scene")

        print(f"\n" + "=" * 50)
        print("ALIGNMENT COMPLETED SUCCESSFULLY!")
        print("=" * 50)

        return alignment_result

    except FileNotFoundError as e:
        print(f"File Error: {e}")
        print("Make sure the reference camera centers file exists.")
        return None

    except ImportError as e:
        print(f"Import Error: {e}")
        print("Please check that all required supporting functions are available:")
        print("- CoreFunctions/Estimate.py (contains estimate)")
        print("- CoreFunctions/Align3d.py (contains align3d)")
        print("- OutputFunctions/DrawScene.py (contains drawscene)")
        print("- OutputFunctions/SaveCalPar.py (contains savecalpar)")
        return None

    except Exception as e:
        print(f"Unexpected Error: {e}")
        print("Please check your input data format and configuration.")
        return None


def usage_notes():
    """
    Print detailed usage instructions for the alignment function.
    """
    print("\n" + "=" * 60)
    print("=== USAGE NOTES FOR YOUR OWN DATA ===")
    print("=" * 60)

    print("\nTo use align_existing_camera_centers() with your own data:")

    print("\n1. PREPARE INPUT DATA DICTIONARY:")
    print("   input_data = {")
    print("       'Cst': camera_centers,        # Shape: (3, N_cameras)")
    print("       'Rot': camera_rotations,      # Shape: (N_cameras, 3, 3)")
    print("       'Pe': projection_matrices,    # List of (3, 4) matrices")
    print("       'Xe': points_3d               # Shape: (3, N_points)")
    print("   }")

    print("\n2. CREATE CONFIGURATION:")
    print("   from CommonCfgAndIO.CalibrationSystem import CalibrationConfig")
    print("   config = CalibrationConfig(")
    print("       data_path='/path/to/your/data/',")
    print("       cameras_2_use=[0, 1, 2, ...]")
    print("   )")

    print("\n3. ENSURE FILES EXIST:")
    print("   - Create 'original_cam_centers.dat' in your data directory")
    print("   - Format: N_cameras rows x 3 columns (X, Y, Z coordinates)")
    print("   - Use: np.savetxt('original_cam_centers.dat', reference_positions)")

    print("\n4. RUN ALIGNMENT:")
    print("   result = align_existing_camera_centers(input_data, config)")

    print("\n5. ACCESS RESULTS:")
    print("   - result['simT']: Similarity transformation parameters (s, R, t)")
    print("   - result['Cst']: Aligned camera centers")
    print("   - result['X']: Aligned 3D points")
    print("   - result['P']: Aligned projection matrices")


if __name__ == "__main__":
    result = main()
    usage_notes()

    print(f"\nExample execution complete!")
    if result is not None:
        print(f"Alignment successful - check the output files!")
    else:
        print(f"Alignment failed - check error messages above.")