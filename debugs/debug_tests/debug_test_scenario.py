#!/usr/bin/env python3

import numpy as np
from pathlib import Path
from phonproj.modes import PhononModes

# Path to BaTiO3 data
BATIO3_YAML_PATH = Path(__file__).parent / "data" / "BaTiO3_phonopy_params.yaml"


def debug_test_scenario():
    """Debug the exact scenario from the failing test."""

    # Generate qpoints for 2x2x2 grid (same as test)
    qpoints_2x2x2 = []
    for i in range(2):
        for j in range(2):
            for k in range(2):
                qpoints_2x2x2.append([i / 2.0, j / 2.0, k / 2.0])
    qpoints_2x2x2 = np.array(qpoints_2x2x2)

    print("Generated q-points:")
    for i, qpoint in enumerate(qpoints_2x2x2):
        print(f"  {i}: {qpoint}")

    # Load phonon data
    modes = PhononModes.from_phonopy_yaml(str(BATIO3_YAML_PATH), qpoints_2x2x2)

    # Find first non-Gamma q-point (same logic as test)
    non_gamma_index = None
    for i, qpoint in enumerate(modes.qpoints):
        if not np.allclose(qpoint, [0, 0, 0], atol=1e-6):
            non_gamma_index = i
            break

    if non_gamma_index is None:
        print("No non-Gamma q-points found!")
        return

    print(
        f"\nFirst non-Gamma q-point: index {non_gamma_index}, qpoint = {modes.qpoints[non_gamma_index]}"
    )

    # Use 2x2x2 supercell (same as test)
    supercell_matrix = np.eye(3, dtype=int) * 2

    # Generate displacements for all modes (same as test)
    all_displacements = modes.generate_all_mode_displacements(
        non_gamma_index, supercell_matrix, amplitude=1.0
    )

    n_modes = all_displacements.shape[0]
    print(f"\nGenerated {n_modes} modes for q-point {modes.qpoints[non_gamma_index]}")

    # Check orthogonality between all pairs
    print("\nOrthogonality check:")
    max_projection = 0
    worst_pair = None

    for i in range(n_modes):
        for j in range(i + 1, n_modes):
            projection = modes.mass_weighted_projection(
                all_displacements[i], all_displacements[j]
            )
            if abs(projection) > max_projection:
                max_projection = abs(projection)
                worst_pair = (i, j)

            if abs(projection) > 1e-6:
                print(f"  FAILED: modes {i}, {j}: projection = {projection}")
            else:
                print(f"  OK:     modes {i}, {j}: projection = {projection}")

    print(f"\nWorst orthogonality violation: {max_projection} for modes {worst_pair}")
    print(f"Test would {'PASS' if max_projection < 1e-6 else 'FAIL'}")

    # Let's also check norms
    print(f"\nNorm check:")
    for i in range(n_modes):
        norm = modes.mass_weighted_norm(all_displacements[i])
        print(f"  Mode {i}: norm = {norm}")


if __name__ == "__main__":
    debug_test_scenario()
