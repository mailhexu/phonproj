#!/usr/bin/env python3

import numpy as np
from pathlib import Path
from phonproj.modes import PhononModes


def compare_methods():
    """Compare phonopy vs original method for a few specific modes."""

    # Test data path
    BATIO3_YAML_PATH = Path("data/BaTiO3_phonopy_params.yaml")

    # Simple test with just first q-point
    qpoints = np.array([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]])

    modes = PhononModes.from_phonopy_yaml(str(BATIO3_YAML_PATH), qpoints)

    # Use 2x2x2 supercell
    supercell_matrix = np.eye(3, dtype=int) * 2

    print("=== COMPARING PHONOPY VS ORIGINAL METHOD ===")

    # Test a few modes from q-point 0 and 1
    test_cases = [(0, 3), (0, 4), (1, 3), (1, 4)]

    for q_index, mode_index in test_cases:
        print(f"\n--- Q-point {q_index}, Mode {mode_index} ---")

        # Try phonopy method
        try:
            phonopy_disp = modes._calculate_supercell_displacements_phonopy(
                q_index=q_index,
                mode_index=mode_index,
                supercell_matrix=supercell_matrix,
                amplitude=1.0,
            )
            phonopy_norm = modes.mass_weighted_norm(phonopy_disp)
            print(f"Phonopy method: norm = {phonopy_norm:.6f}")

            if phonopy_norm < 1e-10:
                print("  ⚠️  PHONOPY GIVES ZERO NORM!")
        except Exception as e:
            print(f"Phonopy method failed: {e}")
            phonopy_disp = None
            phonopy_norm = 0.0

        # Try original method by temporarily modifying the function
        # Let's generate it manually using the original approach
        try:
            from phonproj.core.supercell import generate_supercell, _get_displacements

            frequency, eigenvector = modes.get_mode(q_index, mode_index)
            qpoint = modes.qpoints[q_index]

            supercell = generate_supercell(modes.primitive_cell, supercell_matrix)
            n_cells = len(supercell) // len(modes.primitive_cell)

            original_disp_complex = _get_displacements(
                eigvec=eigenvector,
                q=qpoint,
                amplitude=1.0,
                argument=0.0,
                supercell=supercell,
                mod_func=lambda x: x % (2 * np.pi),
                use_isotropy_amplitude=False,
                normalize=True,
                n_cells=n_cells,
            )
            original_disp = original_disp_complex.real
            original_norm = modes.mass_weighted_norm(original_disp)
            print(f"Original method: norm = {original_norm:.6f}")

            if phonopy_disp is not None and original_norm > 1e-10:
                # Compare the two methods
                projection = abs(
                    modes.mass_weighted_projection(phonopy_disp, original_disp)
                )
                print(f"Cross-projection: {projection:.6f}")
                if projection > 0.99:
                    print("  ✓ Methods agree")
                else:
                    print("  ⚠️  Methods disagree!")

        except Exception as e:
            print(f"Original method failed: {e}")


if __name__ == "__main__":
    compare_methods()
