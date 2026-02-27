"""
Simple Example - How to Use the Calibration System
===================================================
"""

from CommonCfgAndIO.CalibrationSystem import CalibrationSystem


def main():
    print("=" * 60)
    print("MULTI-CAMERA CALIBRATION - SIMPLE EXAMPLE")
    print("=" * 60)

    system = CalibrationSystem()

    # Step 1: Load configuration from paths file
    print("\n1️⃣  Reading configuration...")
    config = system.load_config_from_file('config_paths.txt')

    print(f"   ✅ Config: {config.basename}")
    print(f"   📁 Data: {config.data_path}")
    print(f"   📷 Cameras: {config.num_cameras}")

    # Step 2: Load calibration data
    print("\n2️⃣  Loading data...")
    data = system.load_data(config)

    print(f"   ✅ Points: {data.points.shape}")
    print(f"   ✅ ID Matrix: {data.id_matrix.shape}")

    # Step 3: Show summary
    print("\n3️⃣  Configuration Summary:")
    system.print_config_summary(config)

    # Step 4: Save results
    print("\n4️⃣  Saving results...")
    system.save_results(config, data, 'Results')

    print("\n✅ DONE! Results saved to: Results/")


if __name__ == "__main__":
    main()