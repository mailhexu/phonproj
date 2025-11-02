#!/usr/bin/env python3

import numpy as np
from pathlib import Path
from phonproj.modes import PhononModes

# Path to BaTiO3 data
BATIO3_YAML_PATH = Path(__file__).parent / "data" / "BaTiO3_phonopy_params.yaml"


def test_gauge_effect_on_orthogonality():
    """Test how gauge choice affects orthogonality."""

    # Generate qpoints for 2x2x2 grid
    qpoints_2x2x2 = []
    for i in range(2):
        for j in range(2):
            for k in range(2):
                qpoints_2x2x2.append([i / 2.0, j / 2.0, k / 2.0])
    qpoints_2x2x2 = np.array(qpoints_2x2x2)

    # Load phonon data with different gauges
    for gauge in ["R", "r"]:
        print(f"\n=== GAUGE: {gauge} ===")

        modes = PhononModes.from_phonopy_yaml(str(BATIO3_YAML_PATH), qpoints_2x2x2)
        # Change gauge after loading
        modes.gauge = gauge

        # Use first non-Gamma q-point
        q_index = 1  # [0,0,0.5]

        # Get displacements for problematic mode pairs
        problem_pairs = [(2, 3), (9, 10)]

        displacements = []
        for mode_i in [2, 3, 9, 10]:
            disp = modes.get_eigen_displacement(q_index, mode_i, normalize=True)
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

    # Test with complex eigenvectors (no gauge transformation)
    print(f"\n=== COMPLEX EIGENVECTORS (NO GAUGE) ===")
    modes = PhononModes.from_phonopy_yaml(str(BATIO3_YAML_PATH), qpoints_2x2x2)
    q_index = 1
    problem_pairs = [(2, 3), (9, 10)]

    # Get raw complex eigenvectors
    eigenvectors = modes.eigenvectors[q_index]  # Shape: (n_modes, n_atoms * 3)
    mass_weights = np.sqrt(modes.atomic_masses)
    mass_weights_repeated = np.repeat(mass_weights, 3)

    displacements_complex = []
    for mode_i in [2, 3, 9, 10]:
        # Get complex eigenvector and apply mass weighting
        eigvec = eigenvectors[mode_i].reshape(modes._n_atoms, 3)
        disp = eigvec / mass_weights[:, np.newaxis]

        # Normalize with complex norm
        norm = np.sqrt(np.sum(np.abs(disp) ** 2 / mass_weights[:, np.newaxis] ** 2))
        if norm > 1e-12:
            disp = disp / norm

        displacements_complex.append(disp)

    # Check orthogonality for complex displacements
    for pair_idx, (mode_i, mode_j) in enumerate(problem_pairs):
        local_i = [2, 3, 9, 10].index(mode_i)
        local_j = [2, 3, 9, 10].index(mode_j)

        # Complex mass-weighted inner product
        disp_i = displacements_complex[local_i]
        disp_j = displacements_complex[local_j]

        projection = np.sum(np.conj(disp_i) * disp_j / mass_weights[:, np.newaxis])

        status = "OK" if abs(projection) < 1e-6 else "FAILED"
        print(f"  {status}: Modes {mode_i},{mode_j}: projection = {projection}")


if __name__ == "__main__":
    test_gauge_effect_on_orthogonality()
