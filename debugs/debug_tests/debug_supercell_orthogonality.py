#!/usr/bin/env python3
"""
Debug script to understand supercell orthogonality issues.
"""

import numpy as np
from pathlib import Path
from phonproj.modes import PhononModes

# Load BaTiO3 data
BATIO3_YAML_PATH = Path("data/BaTiO3_phonopy_params.yaml")


def debug_supercell_orthogonality():
    """Debug orthogonality issues with supercell displacements."""

    # Load data with 2x2x2 q-points
    qpoints_2x2x2 = []
    for i in range(2):
        for j in range(2):
            for k in range(2):
                qpoints_2x2x2.append([i / 2.0, j / 2.0, k / 2.0])
    qpoints_2x2x2 = np.array(qpoints_2x2x2)

    modes = PhononModes.from_phonopy_yaml(str(BATIO3_YAML_PATH), qpoints_2x2x2)

    # Test 1x1x1 vs 2x2x2 supercell for same q-point
    supercell_1x1x1 = np.eye(3, dtype=int)
    supercell_2x2x2 = np.eye(3, dtype=int) * 2

    # Find non-Gamma q-point
    non_gamma_index = None
    for i, qpoint in enumerate(modes.qpoints):
        if not np.allclose(qpoint, [0, 0, 0], atol=1e-6):
            non_gamma_index = i
            break

    if non_gamma_index is None:
        print("No non-Gamma q-points found!")
        return

    print(f"Using q-point {non_gamma_index}: {modes.qpoints[non_gamma_index]}")

    # Generate displacements using both methods
    print("\n=== 1x1x1 supercell (should be orthogonal) ===")
    displacements_1x1x1 = modes.generate_all_mode_displacements(
        non_gamma_index, supercell_1x1x1, amplitude=1.0
    )

    # Check orthogonality for 1x1x1
    n_modes = min(5, displacements_1x1x1.shape[0])  # Check first 5 modes
    max_projection_1x1x1 = 0.0
    for i in range(n_modes):
        for j in range(i + 1, n_modes):
            projection = modes.mass_weighted_projection(
                displacements_1x1x1[i], displacements_1x1x1[j]
            )
            max_projection_1x1x1 = max(max_projection_1x1x1, abs(projection))
            if abs(projection) > 1e-10:
                print(f"1x1x1 modes {i},{j}: projection = {projection}")

    print(f"1x1x1 max projection: {max_projection_1x1x1}")

    print("\n=== 2x2x2 supercell (currently not orthogonal) ===")
    displacements_2x2x2 = modes.generate_all_mode_displacements(
        non_gamma_index, supercell_2x2x2, amplitude=1.0
    )

    # Check orthogonality for 2x2x2
    max_projection_2x2x2 = 0.0
    for i in range(n_modes):
        for j in range(i + 1, n_modes):
            projection = modes.mass_weighted_projection(
                displacements_2x2x2[i], displacements_2x2x2[j]
            )
            max_projection_2x2x2 = max(max_projection_2x2x2, abs(projection))
            if abs(projection) > 1e-10:
                print(f"2x2x2 modes {i},{j}: projection = {projection}")

    print(f"2x2x2 max projection: {max_projection_2x2x2}")

    # For 2x2x2, need supercell masses
    supercell_masses = np.tile(modes.atomic_masses, 8)  # 2x2x2 = 8 replicas

    # Check norms
    print("\n=== Displacement Norms ===")
    for i in range(min(3, n_modes)):
        norm_1x1x1 = modes.mass_weighted_norm(displacements_1x1x1[i])
        norm_2x2x2 = modes.mass_weighted_norm(displacements_2x2x2[i], supercell_masses)

        print(f"Mode {i}: 1x1x1 norm = {norm_1x1x1:.6f}, 2x2x2 norm = {norm_2x2x2:.6f}")
        print(f"  Expected 2x2x2 norm for orthonormality: 1/√8 = {1 / np.sqrt(8):.6f}")

    # Debug: Check what _calculate_supercell_displacements produces directly
    print("\n=== Direct supercell method output ===")
    raw_displacement = modes._calculate_supercell_displacements(
        non_gamma_index, 0, supercell_2x2x2, 8 * modes._n_atoms, phase=0.0
    )
    raw_norm = modes.mass_weighted_norm(raw_displacement, supercell_masses)
    print(f"Raw supercell displacement norm: {raw_norm:.6f}")

    # Compare with normalized version
    if raw_norm > 1e-12:
        normalized_displacement = raw_displacement / raw_norm
        normalized_norm = modes.mass_weighted_norm(
            normalized_displacement, supercell_masses
        )
        print(f"After normalization: {normalized_norm:.6f}")


if __name__ == "__main__":
    debug_supercell_orthogonality()
