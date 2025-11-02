#!/usr/bin/env python3
"""
Test the phonopy implementation of supercell displacement calculation.
"""

import numpy as np
import sys

sys.path.insert(0, ".")

from phonproj.modes import PhononModes


def test_phonopy_implementation():
    """Test the new phonopy-based displacement calculation."""
    print("Testing phonopy implementation...")

    # Load test data
    try:
        # Use BaTiO3 data like other tests
        BATIO3_YAML_PATH = "data/BaTiO3_phonopy_params.yaml"

        # Define test q-points (start with Gamma only to avoid force constants issues)
        qpoints = np.array(
            [
                [0.0, 0.0, 0.0],  # Gamma point
            ]
        )

        modes = PhononModes.from_phonopy_yaml(BATIO3_YAML_PATH, qpoints=qpoints)
        print(
            f"✓ Successfully loaded {modes.n_qpoints} q-points with {modes.n_modes} modes each"
        )

        # Test supercell matrix
        supercell_matrix = np.array([[2, 0, 0], [0, 2, 0], [0, 0, 1]])
        print(f"✓ Using supercell matrix: {supercell_matrix.tolist()}")

        # Test a few modes
        test_indices = [(0, 0), (0, 2), (0, 5)]  # (q_index, mode_index) pairs

        for q_idx, mode_idx in test_indices:
            try:
                # Test the new phonopy method
                displacements = modes._calculate_supercell_displacements_phonopy(
                    q_index=q_idx,
                    mode_index=mode_idx,
                    supercell_matrix=supercell_matrix,
                    amplitude=0.1,
                )

                print(
                    f"✓ q={q_idx}, mode={mode_idx}: displacement shape={displacements.shape}, "
                    f"norm={np.linalg.norm(displacements):.6f}"
                )

                # Check if displacements are real (for R gauge)
                if np.any(np.iscomplex(displacements)):
                    print(f"  ⚠ Warning: Complex displacements found")

            except Exception as e:
                print(f"✗ Failed for q={q_idx}, mode={mode_idx}: {e}")
                return False

        print("✓ All tests passed!")
        return True

    except Exception as e:
        print(f"✗ Test failed: {e}")
        return False


if __name__ == "__main__":
    success = test_phonopy_implementation()
    sys.exit(0 if success else 1)
