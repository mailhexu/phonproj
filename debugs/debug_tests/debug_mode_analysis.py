#!/usr/bin/env python3

import numpy as np
from pathlib import Path
from phonproj.modes import PhononModes

# Path to BaTiO3 data
BATIO3_YAML_PATH = Path(__file__).parent / "data" / "BaTiO3_phonopy_params.yaml"


def analyze_modes():
    """Analyze which modes have issues."""

    # Generate qpoints for 2x2x2 grid (same as test)
    qpoints_2x2x2 = []
    for i in range(2):
        for j in range(2):
            for k in range(2):
                qpoints_2x2x2.append([i / 2.0, j / 2.0, k / 2.0])
    qpoints_2x2x2 = np.array(qpoints_2x2x2)

    # Load phonon data
    modes = PhononModes.from_phonopy_yaml(str(BATIO3_YAML_PATH), qpoints_2x2x2)

    # Find first non-Gamma q-point (should be [0,0,0.5])
    non_gamma_index = 1  # We know from debug this is [0,0,0.5]

    print(f"Analyzing q-point {non_gamma_index}: {modes.qpoints[non_gamma_index]}")
    print(f"Frequencies: {modes.frequencies[non_gamma_index]}")
    print()

    # Use 2x2x2 supercell
    supercell_matrix = np.eye(3, dtype=int) * 2

    # Generate displacements for all modes
    all_displacements = modes.generate_all_mode_displacements(
        non_gamma_index, supercell_matrix, amplitude=1.0
    )

    n_modes = all_displacements.shape[0]

    # Analyze each mode
    print("Mode analysis:")
    for i in range(n_modes):
        norm = modes.mass_weighted_norm(all_displacements[i])
        freq = modes.frequencies[non_gamma_index, i]
        print(f"  Mode {i:2d}: freq = {freq:8.3f} cm⁻¹, norm = {norm:.6e}")

    print()

    # Identify problematic modes
    tiny_norm_modes = []
    normal_modes = []

    for i in range(n_modes):
        norm = modes.mass_weighted_norm(all_displacements[i])
        if norm < 1e-10:
            tiny_norm_modes.append(i)
        else:
            normal_modes.append(i)

    print(f"Modes with tiny norms (< 1e-10): {tiny_norm_modes}")
    print(f"Modes with normal norms: {normal_modes}")
    print()

    # Check orthogonality within each group
    print("Orthogonality within normal modes:")
    for i in range(len(normal_modes)):
        for j in range(i + 1, len(normal_modes)):
            mode_i = normal_modes[i]
            mode_j = normal_modes[j]
            projection = modes.mass_weighted_projection(
                all_displacements[mode_i], all_displacements[mode_j]
            )
            status = "OK" if abs(projection) < 1e-6 else "FAILED"
            print(
                f"  {status}: modes {mode_i:2d}, {mode_j:2d}: projection = {projection:.6e}"
            )


if __name__ == "__main__":
    analyze_modes()
