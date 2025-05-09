import numpy as np
import os


def readradfile(name):
    """
    Reads BlueC *.rad files containing camera calibration parameters

    The .rad files store the camera calibration matrix K and
    distortion coefficients kc in a specific format.

    Parameters:
    -----------
    name : str
        Full path to the .rad file

    Returns:
    --------
    K : numpy.ndarray (3x3)
        Camera calibration matrix
    kc : numpy.ndarray (4x1)
        Vector of distortion parameters [k1, k2, p1, p2]
        where k1, k2 are radial distortion coefficients and
        p1, p2 are tangential distortion coefficients

    Raises:
    -------
    FileNotFoundError : If the file cannot be opened

    Notes:
    ------
    The .rad file format is specific to the BlueC calibration system.
    It contains the 3x3 calibration matrix followed by 4 distortion coefficients.
    """

    # Check if file exists
    if not os.path.isfile(name):
        raise FileNotFoundError(f"Could not open {name}. Missing rad files?")

    # Initialize matrices
    K = np.zeros((3, 3))
    kc = np.zeros(4)

    # Read the file
    with open(name, 'r') as f:
        lines = f.readlines()

        # Process the first 9 lines for the calibration matrix K
        line_idx = 0
        for i in range(3):
            for j in range(3):
                # Extract the value part, starting from position 7
                line = lines[line_idx].strip()
                str_end = line[7:]

                # Remove trailing semicolon if present
                if str_end.endswith(';'):
                    str_end = str_end[:-1]

                # Convert to float and store in K
                K[i, j] = float(str_end)
                line_idx += 1

        # Skip one line
        line_idx += 1

        # Process the next 4 lines for distortion coefficients
        for i in range(4):
            line = lines[line_idx].strip()
            str_end = line[7:]

            # Remove trailing semicolon if present
            if str_end.endswith(';'):
                str_end = str_end[:-1]

            # Convert to float and store in kc
            kc[i] = float(str_end)
            line_idx += 1

    return K, kc