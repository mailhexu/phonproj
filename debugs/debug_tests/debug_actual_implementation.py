#!/usr/bin/env python3

import numpy as np
from pathlib import Path
from phonproj.modes import PhononModes

# Path to BaTiO3 data
BATIO3_YAML_PATH = Path(__file__).parent / "data" / "BaTiO3_phonopy_params.yaml"


def test_actual_implementation():
    """Test what the actual implementation generates vs direct method."""

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

    print(
        f"Testing actual implementation for q-point {q_index}: {modes.qpoints[q_index]}"
    )

    # Test both 1x1x1 and 2x2x2 supercells using the actual implementation
    supercell_matrices = [
        np.eye(3, dtype=int),  # 1x1x1 (uses get_eigen_displacement)
        np.eye(3, dtype=int) * 2,  # 2x2x2 (uses direct method)
    ]

    problem_pairs = [(2, 3), (9, 10)]

    for i, supercell_matrix in enumerate(supercell_matrices):
        size_name = "1x1x1" if i == 0 else "2x2x2"
        print(f"\n=== {size_name} SUPERCELL (ACTUAL IMPLEMENTATION) ===")

        # Use the actual generate_all_mode_displacements method
        all_displacements = modes.generate_all_mode_displacements(
            q_index, supercell_matrix, amplitude=1.0
        )

        # Extract the problematic modes
        displacements = [all_displacements[mode_i] for mode_i in [2, 3, 9, 10]]

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

    # Also test get_eigen_displacement directly for comparison
    print(f"\n=== DIRECT get_eigen_displacement METHOD ===")
    displacements_direct = []
    for mode_i in [2, 3, 9, 10]:
        disp = modes.get_eigen_displacement(q_index, mode_i, normalize=True)
        displacements_direct.append(disp)

    # Check norms for direct method
    for j, mode_i in enumerate([2, 3, 9, 10]):
        norm = modes.mass_weighted_norm(displacements_direct[j])
        print(f"  Mode {mode_i}: norm = {norm:.6e}")

    # Check orthogonality for direct method
    for pair_idx, (mode_i, mode_j) in enumerate(problem_pairs):
        local_i = [2, 3, 9, 10].index(mode_i)
        local_j = [2, 3, 9, 10].index(mode_j)
        projection = modes.mass_weighted_projection(
            displacements_direct[local_i], displacements_direct[local_j]
        )
        status = "OK" if abs(projection) < 1e-6 else "FAILED"
        print(f"  {status}: Modes {mode_i},{mode_j}: projection = {projection:.6e}")


if __name__ == "__main__":
    test_actual_implementation()
