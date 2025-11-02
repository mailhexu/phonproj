#!/usr/bin/env python3
"""
Test different mass-weighted inner product conventions to find the correct one
for Phonopy eigenvectors.
"""

import numpy as np
from pathlib import Path
from phonproj.modes import PhononModes


def test_mass_conventions():
    """Test different mass-weighted inner product conventions."""

    # Load BaTiO3 data
    BATIO3_YAML_PATH = Path("data/BaTiO3_phonopy_params.yaml")
    gamma_qpoint = np.array([[0.0, 0.0, 0.0]])
    modes = PhononModes.from_phonopy_yaml(str(BATIO3_YAML_PATH), gamma_qpoint)

    eigenvectors = modes.eigenvectors[0]  # Shape: (n_modes, n_atoms*3)
    masses = modes.atomic_masses
    masses_repeated = np.repeat(masses, 3)  # Shape: (n_atoms*3,)

    print("=== Testing Mass-Weighted Inner Product Conventions ===")
    print(f"Eigenvector shape: {eigenvectors.shape}")
    print(f"Masses shape: {masses.shape}")
    print(f"Repeated masses shape: {masses_repeated.shape}")
    print()

    # Test different conventions
    conventions = [
        (
            "Current: m * u1* · u2",
            lambda u1, u2: np.sum(masses_repeated * np.conj(u1) * u2),
        ),
        (
            "1/m * u1* · u2",
            lambda u1, u2: np.sum((1 / masses_repeated) * np.conj(u1) * u2),
        ),
        (
            "1/√m * u1* · u2",
            lambda u1, u2: np.sum((1 / np.sqrt(masses_repeated)) * np.conj(u1) * u2),
        ),
        (
            "√m * u1* · u2",
            lambda u1, u2: np.sum(np.sqrt(masses_repeated) * np.conj(u1) * u2),
        ),
        ("Plain: u1* · u2", lambda u1, u2: np.sum(np.conj(u1) * u2)),
    ]

    for name, inner_product in conventions:
        print(f"=== Convention: {name} ===")

        # Check orthonormality
        ortho_matrix = np.zeros((15, 15), dtype=complex)
        for i in range(15):
            for j in range(15):
                ortho_matrix[i, j] = inner_product(eigenvectors[i], eigenvectors[j])

        # Analyze
        diagonal = np.diag(ortho_matrix)
        identity = np.eye(15)
        max_deviation = np.max(np.abs(ortho_matrix - identity))

        print(
            f"  Diagonal range: [{np.min(diagonal.real):.6f}, {np.max(diagonal.real):.6f}]"
        )
        print(
            f"  Max off-diagonal: {np.max(np.abs(ortho_matrix - np.diag(diagonal))):.6f}"
        )
        print(f"  Max deviation from identity: {max_deviation:.6f}")

        # Check if this gives orthonormality
        if max_deviation < 1e-10:
            print(f"  ✓ ORTHONORMAL under this convention!")
        print()


if __name__ == "__main__":
    test_mass_conventions()
