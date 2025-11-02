#!/usr/bin/env python3

import numpy as np
from pathlib import Path
from phonproj.modes import PhononModes

# Path to BaTiO3 data
BATIO3_YAML_PATH = Path(__file__).parent / "data" / "BaTiO3_phonopy_params.yaml"


def check_raw_eigenvector_orthogonality():
    """Check if the raw eigenvectors from phonopy are orthogonal."""

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
        f"Checking raw eigenvector orthogonality for q-point {q_index}: {modes.qpoints[q_index]}"
    )
    print(f"Frequencies: {modes.frequencies[q_index]}")
    print()

    # Get raw eigenvectors
    eigenvectors = modes.eigenvectors[q_index]  # Shape: (n_modes, n_atoms * 3)

    # Check orthogonality with mass weighting
    mass_weights = np.sqrt(modes.atomic_masses)
    mass_weights_repeated = np.repeat(mass_weights, 3)  # Repeat for x,y,z

    print("Raw eigenvector orthogonality (mass-weighted):")
    problem_pairs = [(2, 3), (9, 10)]

    for mode_i, mode_j in problem_pairs:
        # Get eigenvectors
        eigvec_i = eigenvectors[mode_i]
        eigvec_j = eigenvectors[mode_j]

        # Mass-weighted inner product
        mass_weighted_inner_product = np.sum(
            np.conj(eigvec_i) * eigvec_j / mass_weights_repeated
        )

        print(
            f"  Modes {mode_i},{mode_j}: mass-weighted inner product = {mass_weighted_inner_product}"
        )
        print(f"    Real part: {mass_weighted_inner_product.real}")
        print(f"    Imag part: {mass_weighted_inner_product.imag}")
        print(f"    Magnitude: {abs(mass_weighted_inner_product)}")
        print()

    print("Raw eigenvector orthogonality (unweighted):")
    for mode_i, mode_j in problem_pairs:
        # Get eigenvectors
        eigvec_i = eigenvectors[mode_i]
        eigvec_j = eigenvectors[mode_j]

        # Unweighted inner product
        inner_product = np.sum(np.conj(eigvec_i) * eigvec_j)

        print(f"  Modes {mode_i},{mode_j}: unweighted inner product = {inner_product}")
        print(f"    Real part: {inner_product.real}")
        print(f"    Imag part: {inner_product.imag}")
        print(f"    Magnitude: {abs(inner_product)}")
        print()


if __name__ == "__main__":
    check_raw_eigenvector_orthogonality()
