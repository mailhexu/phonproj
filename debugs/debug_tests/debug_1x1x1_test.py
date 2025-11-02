#!/usr/bin/env python3

import numpy as np
from pathlib import Path
from phonproj.modes import PhononModes

# Path to BaTiO3 data
BATIO3_YAML_PATH = Path(__file__).parent / "data" / "BaTiO3_phonopy_params.yaml"


def test_1x1x1_orthogonality():
    """Test orthogonality for 1x1x1 supercell (should be perfect)."""

    # Generate qpoints for 2x2x2 grid
    qpoints_2x2x2 = []
    for i in range(2):
        for j in range(2):
            for k in range(2):
                qpoints_2x2x2.append([i / 2.0, j / 2.0, k / 2.0])
    qpoints_2x2x2 = np.array(qpoints_2x2x2)

    # Load phonon data
    modes = PhononModes.from_phonopy_yaml(str(BATIO3_YAML_PATH), qpoints_2x2x2)

    # Use first non-Gamma q-point
    q_index = 1  # [0,0,0.5]

    print(f"Testing 1x1x1 supercell for q-point {q_index}: {modes.qpoints[q_index]}")

    # Test both 1x1x1 and 2x2x2 supercells
    supercell_matrices = [
        np.eye(3, dtype=int),  # 1x1x1
        np.eye(3, dtype=int) * 2,  # 2x2x2
    ]

    problem_pairs = [(2, 3), (9, 10)]

    for i, supercell_matrix in enumerate(supercell_matrices):
        size_name = "1x1x1" if i == 0 else "2x2x2"
        print(f"\n=== {size_name} SUPERCELL ===")

        # Generate displacements using direct method with r gauge
        displacements = []
        for mode_i in [2, 3, 9, 10]:
            disp = modes._calculate_supercell_displacements_direct(
                q_index, mode_i, supercell_matrix, amplitude=1.0, gauge="r"
            )
            displacements.append(disp)

        # Check norms
        for j, mode_i in enumerate([2, 3, 9, 10]):
            norm = modes.mass_weighted_norm(displacements[j])
            print(f"  Mode {mode_i}: norm = {norm:.6e}")

        # Check orthogonality for each problematic pair
        for pair_idx, (mode_i, mode_j) in enumerate(problem_pairs):
            local_i = [2, 3, 9, 10].index(mode_i)
            local_j = [2, 3, 9, 10].index(mode_j)
            projection = modes.mass_weighted_projection(
                displacements[local_i], displacements[local_j]
            )
            status = "OK" if abs(projection) < 1e-6 else "FAILED"
            print(f"  {status}: Modes {mode_i},{mode_j}: projection = {projection:.6e}")


if __name__ == "__main__":
    test_1x1x1_orthogonality()
