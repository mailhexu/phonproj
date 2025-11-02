#!/usr/bin/env python3

import numpy as np
from pathlib import Path
from phonproj.modes import PhononModes


def test_phonopy_modulation():
    """Test the phonopy modulation API implementation."""

    # Test data path
    BATIO3_YAML_PATH = Path("data/BaTiO3_phonopy_params.yaml")

    # Simple test with just first q-point
    qpoints = np.array([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]])

    modes = PhononModes.from_phonopy_yaml(str(BATIO3_YAML_PATH), qpoints)

    # Use 2x2x2 supercell
    supercell_matrix = np.eye(3, dtype=int) * 2

    print("=== TESTING PHONOPY MODULATION API ===")

    # Test a few modes
    test_cases = [(0, 3), (0, 4), (1, 3)]

    for q_index, mode_index in test_cases:
        print(f"\n--- Q-point {q_index}, Mode {mode_index} ---")

        try:
            # Use phonopy modulation API
            phonopy_disp = modes._calculate_supercell_displacements_phonopy(
                q_index=q_index,
                mode_index=mode_index,
                supercell_matrix=supercell_matrix,
                amplitude=1.0,
            )

            print(f"Success! Shape: {phonopy_disp.shape}")

            # Check norm before and after our normalization
            # For supercell, we need supercell masses
            supercell = modes.generate_supercell(supercell_matrix)
            supercell_masses = supercell.get_masses()
            raw_norm = modes.mass_weighted_norm(phonopy_disp, supercell_masses)
            print(f"Raw norm from phonopy: {raw_norm:.6f}")

            # Apply our normalization
            if raw_norm > 1e-12:
                normalized_disp = phonopy_disp * 1.0 / raw_norm
                final_norm = modes.mass_weighted_norm(normalized_disp, supercell_masses)
                print(f"After normalization to 1.0: {final_norm:.6f}")
            else:
                print("⚠️  Zero norm from phonopy!")

        except Exception as e:
            print(f"❌ Phonopy modulation failed: {e}")


if __name__ == "__main__":
    test_phonopy_modulation()
