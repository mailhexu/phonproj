#!/usr/bin/env python3
"""
Test script to compare r gauge vs R gauge implementation
against phonopy API results.
"""

import numpy as np
from pathlib import Path
from phonproj.modes import PhononModes

# Data path
BATIO3_YAML_PATH = Path(__file__).parent / "data" / "BaTiO3_phonopy_params.yaml"


def test_gauge_comparison():
    """Test if r gauge in direct method matches phonopy API."""

    # Set up the same test case as the failing test
    qpoints_2x2x2 = []
    for i in range(2):
        for j in range(2):
            for k in range(2):
                qpoints_2x2x2.append([i / 2.0, j / 2.0, k / 2.0])
    qpoints_2x2x2 = np.array(qpoints_2x2x2)

    modes = PhononModes.from_phonopy_yaml(str(BATIO3_YAML_PATH), qpoints_2x2x2)

    # Use 2x2x2 supercell
    supercell_matrix = np.eye(3, dtype=int) * 2

    # Find a non-Gamma q-point (same as the test)
    non_gamma_index = None
    for i, qpoint in enumerate(modes.qpoints):
        if not np.allclose(qpoint, [0, 0, 0], atol=1e-6):
            non_gamma_index = i
            break

    if non_gamma_index is None:
        print("No non-Gamma q-points found!")
        return

    print(f"Using q-point index {non_gamma_index}: {modes.qpoints[non_gamma_index]}")

    # Test first two modes
    mode_indices = [0, 1]

    print("\n=== PHONOPY API METHOD ===")
    phonopy_displacements = []
    for mode_idx in mode_indices:
        # Use the phonopy API method directly
        displacement = modes._calculate_supercell_displacements_phonopy(
            non_gamma_index, mode_idx, supercell_matrix, amplitude=1.0
        )

        # Apply the same normalization as in generate_all_mode_displacements
        current_norm = modes.mass_weighted_norm(displacement)
        if current_norm > 1e-12:
            displacement = displacement * 1.0 / current_norm

        phonopy_displacements.append(displacement)
        print(f"Mode {mode_idx}: norm = {modes.mass_weighted_norm(displacement):.6f}")

    # Check phonopy orthogonality
    projection_phonopy = modes.mass_weighted_projection(
        phonopy_displacements[0], phonopy_displacements[1]
    )
    print(f"Phonopy orthogonality: projection = {projection_phonopy}")

    print("\n=== DIRECT METHOD - R GAUGE ===")
    r_gauge_displacements = []
    for mode_idx in mode_indices:
        displacement = modes._calculate_supercell_displacements_direct(
            non_gamma_index, mode_idx, supercell_matrix, amplitude=1.0, gauge="R"
        )
        r_gauge_displacements.append(displacement)
        print(f"Mode {mode_idx}: norm = {modes.mass_weighted_norm(displacement):.6f}")

    projection_r_gauge = modes.mass_weighted_projection(
        r_gauge_displacements[0], r_gauge_displacements[1]
    )
    print(f"R gauge orthogonality: projection = {projection_r_gauge}")

    print("\n=== DIRECT METHOD - r GAUGE ===")
    r_gauge_displacements = []
    for mode_idx in mode_indices:
        displacement = modes._calculate_supercell_displacements_direct(
            non_gamma_index, mode_idx, supercell_matrix, amplitude=1.0, gauge="r"
        )
        r_gauge_displacements.append(displacement)
        print(f"Mode {mode_idx}: norm = {modes.mass_weighted_norm(displacement):.6f}")

    projection_r_gauge = modes.mass_weighted_projection(
        r_gauge_displacements[0], r_gauge_displacements[1]
    )
    print(f"r gauge orthogonality: projection = {projection_r_gauge}")

    print("\n=== COMPARISON ===")
    print(f"Phonopy projection:  {projection_phonopy}")
    print(f"R gauge projection:  {projection_r_gauge}")
    print(f"r gauge projection:  {projection_r_gauge}")

    # Compare displacement patterns between phonopy and r gauge
    print(f"\n=== PHONOPY vs r GAUGE COMPARISON ===")
    for mode_idx in range(2):
        diff = phonopy_displacements[mode_idx] - r_gauge_displacements[mode_idx]
        max_diff = np.max(np.abs(diff))
        rms_diff = np.sqrt(np.mean(diff**2))
        print(f"Mode {mode_idx}: max_diff = {max_diff:.6f}, rms_diff = {rms_diff:.6f}")

        # Check correlation
        phonopy_norm = np.linalg.norm(phonopy_displacements[mode_idx])
        r_gauge_norm = np.linalg.norm(r_gauge_displacements[mode_idx])

        if phonopy_norm > 1e-12 and r_gauge_norm > 1e-12:
            correlation = np.dot(
                phonopy_displacements[mode_idx].flatten(),
                r_gauge_displacements[mode_idx].flatten(),
            ) / (phonopy_norm * r_gauge_norm)
            print(f"Mode {mode_idx}: correlation = {correlation:.6f}")

    print(f"\n=== ORTHOGONALITY TEST ===")
    print(f"Phonopy API orthogonal (< 1e-6): {abs(projection_phonopy) < 1e-6}")
    print(f"r gauge orthogonal (< 1e-6):    {abs(projection_r_gauge) < 1e-6}")


if __name__ == "__main__":
    test_gauge_comparison()
