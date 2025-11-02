#!/usr/bin/env python3
"""
Debug the r gauge implementation to see why it produces zero norm.
"""

import numpy as np
from pathlib import Path
from phonproj.modes import PhononModes

# Data path
BATIO3_YAML_PATH = Path(__file__).parent / "data" / "BaTiO3_phonopy_params.yaml"


def debug_r_gauge():
    """Debug r gauge implementation step by step."""

    qpoints_2x2x2 = []
    for i in range(2):
        for j in range(2):
            for k in range(2):
                qpoints_2x2x2.append([i / 2.0, j / 2.0, k / 2.0])
    qpoints_2x2x2 = np.array(qpoints_2x2x2)

    modes = PhononModes.from_phonopy_yaml(str(BATIO3_YAML_PATH), qpoints_2x2x2)

    # Find a non-Gamma q-point
    non_gamma_index = None
    for i, qpoint in enumerate(modes.qpoints):
        if not np.allclose(qpoint, [0, 0, 0], atol=1e-6):
            non_gamma_index = i
            break

    if non_gamma_index is None:
        print("No non-Gamma q-points found!")
        return

    print(f"Q-point: {modes.qpoints[non_gamma_index]}")

    # Debug the r gauge step-by-step for mode 0
    mode_idx = 0
    eigenvector = modes.eigenvectors[non_gamma_index, mode_idx]
    qpoint = modes.qpoints[non_gamma_index]

    print(f"\nOriginal eigenvector (first 6 components): {eigenvector[:6]}")
    print(f"Original eigenvector magnitude: {np.linalg.norm(eigenvector)}")

    # Get primitive cell scaled positions
    scaled_positions = modes.primitive_cell.get_scaled_positions()
    print(f"Scaled positions shape: {scaled_positions.shape}")
    print(f"First few scaled positions: {scaled_positions[:3]}")

    # Apply r gauge: exp(2j*π*q·r) phase factor
    phases = np.exp(2j * np.pi * np.dot(scaled_positions, qpoint))
    print(f"Phases shape: {phases.shape}")
    print(f"Phases (first 3): {phases[:3]}")
    print(f"Phase magnitudes: {np.abs(phases[:3])}")

    # Repeat for each Cartesian direction
    phases_3d = np.repeat(phases, 3)
    print(f"3D phases shape: {phases_3d.shape}")

    # Apply gauge transformation
    gauge_eigenvector = eigenvector * phases_3d
    print(f"Gauge eigenvector magnitude: {np.linalg.norm(gauge_eigenvector)}")
    print(f"Gauge eigenvector (first 6): {gauge_eigenvector[:6]}")

    # Reshape and apply mass weighting
    unit_cell_displacement = gauge_eigenvector.reshape(modes._n_atoms, 3)
    print(f"Unit cell displacement shape: {unit_cell_displacement.shape}")

    # Mass weighting
    mass_weights = np.sqrt(modes.atomic_masses)
    print(f"Mass weights: {mass_weights}")

    unit_cell_displacement = unit_cell_displacement / mass_weights[:, np.newaxis]
    print(f"After mass weighting magnitude: {np.linalg.norm(unit_cell_displacement)}")

    # Take real part
    unit_cell_displacement_real = unit_cell_displacement.real
    print(
        f"After taking real part magnitude: {np.linalg.norm(unit_cell_displacement_real)}"
    )
    print(f"Real displacement (first atom): {unit_cell_displacement_real[0]}")

    # Compare with R gauge
    print(f"\n=== R GAUGE COMPARISON ===")
    r_gauge_eigenvector = eigenvector  # No additional phase
    r_gauge_displacement = r_gauge_eigenvector.reshape(modes._n_atoms, 3)
    r_gauge_displacement = r_gauge_displacement / mass_weights[:, np.newaxis]
    r_gauge_displacement_real = r_gauge_displacement.real
    print(f"R gauge magnitude: {np.linalg.norm(r_gauge_displacement_real)}")
    print(f"R gauge (first atom): {r_gauge_displacement_real[0]}")


if __name__ == "__main__":
    debug_r_gauge()
