#!/usr/bin/env python3
"""
Extended debug script for Step 7 implementation.
"""

import numpy as np
from pathlib import Path
from phonproj.modes import PhononModes

# Test data path
BATIO3_YAML_PATH = Path(__file__).parent / "data" / "BaTiO3_phonopy_params.yaml"


def debug_eigenvectors():
    """Debug the original eigenvectors from phonopy."""

    # Load BaTiO3 data - only need Gamma point
    gamma_qpoint = np.array([[0.0, 0.0, 0.0]])
    print(f"Loading data from {BATIO3_YAML_PATH}")
    modes = PhononModes.from_phonopy_yaml(str(BATIO3_YAML_PATH), gamma_qpoint)

    print(f"Loaded modes with {modes._n_atoms} atoms")
    print(f"Frequencies shape: {modes.frequencies.shape}")
    print(f"Eigenvectors shape: {modes.eigenvectors.shape}")

    q_index = 0
    eigenvectors = modes.eigenvectors[q_index]  # Shape: (n_modes, n_atoms*3)

    print(f"\nChecking original eigenvectors orthogonality:")
    print(f"Eigenvectors shape: {eigenvectors.shape}")

    # Check if the raw eigenvectors are orthonormal
    n_modes = eigenvectors.shape[0]
    orthogonality_matrix = np.zeros((n_modes, n_modes), dtype=complex)

    for i in range(n_modes):
        for j in range(n_modes):
            # Simple dot product (no mass weighting)
            dot_product = np.vdot(eigenvectors[i], eigenvectors[j])
            orthogonality_matrix[i, j] = dot_product

    print(f"Diagonal elements (should be 1): {np.diag(orthogonality_matrix)[:5]}")
    print(f"Off-diagonal max: {np.max(np.abs(orthogonality_matrix - np.eye(n_modes)))}")

    # Check mass-weighted orthogonality of eigenvectors
    print(f"\nChecking mass-weighted orthogonality of eigenvectors:")
    masses = modes.atomic_masses
    masses_repeated = np.repeat(masses, 3)

    mass_orthogonality_matrix = np.zeros((n_modes, n_modes), dtype=complex)

    for i in range(n_modes):
        for j in range(n_modes):
            # Mass-weighted dot product
            mass_dot_product = np.sum(
                masses_repeated * np.conj(eigenvectors[i]) * eigenvectors[j]
            )
            mass_orthogonality_matrix[i, j] = mass_dot_product

    print(f"Mass-weighted diagonal elements: {np.diag(mass_orthogonality_matrix)[:5]}")
    print(
        f"Mass-weighted off-diagonal max: {np.max(np.abs(mass_orthogonality_matrix - np.eye(n_modes)))}"
    )

    # Now check the generated displacements using get_eigen_displacement
    print(f"\nChecking get_eigen_displacement outputs:")

    displacement_matrix = np.zeros((n_modes, n_modes), dtype=complex)

    for i in range(min(5, n_modes)):  # Check first 5 modes
        disp_i = modes.get_eigen_displacement(q_index, i)
        norm_i = modes.mass_weighted_norm(disp_i)
        print(f"Mode {i}: displacement norm = {norm_i}")

        for j in range(min(5, n_modes)):
            disp_j = modes.get_eigen_displacement(q_index, j)
            projection = modes.mass_weighted_projection(disp_i, disp_j)
            displacement_matrix[i, j] = projection

    print(f"get_eigen_displacement mass-weighted matrix (5x5):")
    print(displacement_matrix[:5, :5])


if __name__ == "__main__":
    debug_eigenvectors()
