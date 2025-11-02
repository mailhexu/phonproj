#!/usr/bin/env python3
"""
Debug script to investigate the orthonormality issue in supercell mode displacements.
Focused on the 1x1x1 Gamma point case that's failing.
"""

import numpy as np
from pathlib import Path
from phonproj.modes import PhononModes

# Path to BaTiO3 data
BATIO3_YAML_PATH = Path("data/BaTiO3_phonopy_params.yaml")


def main():
    print("=== Debugging Orthonormality Issue (1x1x1 Gamma Point) ===\n")

    # Load BaTiO3 data - only need Gamma point
    gamma_qpoint = np.array([[0.0, 0.0, 0.0]])
    modes = PhononModes.from_phonopy_yaml(str(BATIO3_YAML_PATH), gamma_qpoint)

    print(f"Number of atoms in unit cell: {modes.n_atoms}")
    print(f"Number of modes: {modes.n_modes}")
    print(f"Atomic masses: {modes.atomic_masses}")
    print()

    # Use 1x1x1 supercell (identity matrix)
    supercell_matrix = np.eye(3, dtype=int)
    gamma_index = 0

    print("=== Step 1: Check raw eigenvector orthonormality ===")
    eigenvectors = modes.eigenvectors[gamma_index]  # Shape: (n_modes, n_atoms, 3)
    print(f"Eigenvector shape: {eigenvectors.shape}")

    # Check raw eigenvector orthonormality
    n_modes = eigenvectors.shape[0]
    raw_ortho_matrix = np.zeros((n_modes, n_modes), dtype=complex)

    for i in range(n_modes):
        for j in range(n_modes):
            # Flatten and compute inner product
            v1 = eigenvectors[i].flatten()
            v2 = eigenvectors[j].flatten()
            raw_ortho_matrix[i, j] = np.dot(np.conj(v1), v2)

    raw_max_dev = np.max(np.abs(raw_ortho_matrix - np.eye(n_modes)))
    print(f"Raw eigenvector orthonormality max deviation: {raw_max_dev}")
    print()

    print("=== Step 2: Check mass-weighted eigenvector orthonormality ===")
    mass_weighted_ortho_matrix = np.zeros((n_modes, n_modes), dtype=complex)

    for i in range(n_modes):
        for j in range(n_modes):
            # Mass-weighted inner product using the modes' method
            projection = modes.mass_weighted_projection(
                eigenvectors[i], eigenvectors[j]
            )
            mass_weighted_ortho_matrix[i, j] = projection

    mass_weighted_max_dev = np.max(np.abs(mass_weighted_ortho_matrix - np.eye(n_modes)))
    print(
        f"Mass-weighted eigenvector orthonormality max deviation: {mass_weighted_max_dev}"
    )
    print()

    print("=== Step 3: Check individual mode displacement generation ===")
    # Test a few individual modes
    for mode_idx in [0, 5, 10]:
        if mode_idx >= n_modes:
            continue

        print(f"--- Mode {mode_idx} ---")

        # Generate displacement using existing method
        displacement = modes.generate_mode_displacement(
            gamma_index, mode_idx, supercell_matrix, amplitude=1.0, normalize=True
        )
        print(f"Displacement shape: {displacement.shape}")

        # Compare with eigenvector
        expected_displacement = eigenvectors[mode_idx]
        print(f"Expected shape: {expected_displacement.shape}")

        # Calculate norms
        mass_weighted_norm = np.sqrt(
            modes.mass_weighted_projection(displacement, displacement).real
        )
        expected_norm = np.sqrt(
            modes.mass_weighted_projection(
                expected_displacement, expected_displacement
            ).real
        )

        print(f"Generated displacement mass-weighted norm: {mass_weighted_norm}")
        print(f"Expected displacement mass-weighted norm: {expected_norm}")

        # Check if they're proportional
        if np.allclose(
            displacement.flatten(), expected_displacement.flatten(), atol=1e-10
        ):
            print("Displacements are identical")
        else:
            ratio = np.linalg.norm(displacement.flatten()) / np.linalg.norm(
                expected_displacement.flatten()
            )
            print(f"Displacement ratio (norm-wise): {ratio}")
        print()

    print("=== Step 4: Check generated displacement orthonormality ===")
    # Generate all displacements using our method
    all_displacements = modes.generate_all_mode_displacements(
        gamma_index, supercell_matrix, amplitude=1.0
    )
    print(f"All displacements shape: {all_displacements.shape}")

    # Check orthonormality
    generated_ortho_matrix = np.zeros((n_modes, n_modes), dtype=complex)

    for i in range(n_modes):
        for j in range(n_modes):
            projection = modes.mass_weighted_projection(
                all_displacements[i], all_displacements[j]
            )
            generated_ortho_matrix[i, j] = projection

    generated_max_dev = np.max(np.abs(generated_ortho_matrix - np.eye(n_modes)))
    print(f"Generated displacement orthonormality max deviation: {generated_max_dev}")

    print("\n=== Step 5: Detailed analysis of problematic entries ===")
    identity = np.eye(n_modes)
    deviations = np.abs(generated_ortho_matrix - identity)
    max_indices = np.unravel_index(np.argmax(deviations), deviations.shape)

    print(f"Maximum deviation at indices {max_indices}: {deviations[max_indices]}")
    print(f"Matrix value at max deviation: {generated_ortho_matrix[max_indices]}")
    print(f"Expected value: {identity[max_indices]}")

    # Show some diagonal and off-diagonal values
    print("\nDiagonal values (should be 1):")
    for i in range(min(5, n_modes)):
        print(f"  [{i},{i}]: {generated_ortho_matrix[i, i]}")

    print("\nOff-diagonal values (should be 0):")
    for i in range(min(3, n_modes)):
        for j in range(min(3, n_modes)):
            if i != j:
                print(f"  [{i},{j}]: {generated_ortho_matrix[i, j]}")


if __name__ == "__main__":
    main()
