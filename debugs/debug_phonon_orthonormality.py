#!/usr/bin/env python3
"""
Debug script to investigate phonon mode orthonormality in BaTiO3.
This will help us understand why sum of projection coefficients > 1.0
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from phonproj.modes import PhononModes


def test_phonon_orthonormality():
    """Test if phonon modes are truly orthonormal in mass-weighted sense"""
    print("=" * 80)
    print("PHONON MODE ORTHONORMALITY ANALYSIS")
    print("=" * 80)

    # Load BaTiO3 data (only Gamma point)
    gamma_qpoint = np.array([[0.0, 0.0, 0.0]])
    modes = PhononModes.from_phonopy_yaml(
        "data/BaTiO3_phonopy_params.yaml", gamma_qpoint
    )

    print(f"Dataset info:")
    print(f"  Q-points: {len(modes.qpoints)}")
    print(f"  Modes per q-point: {modes.frequencies.shape[1]}")
    print(f"  Unit cell atoms: {len(modes.primitive_cell)}")
    print(f"  Frequencies: {modes.frequencies[0]} cm^-1")

    # Get eigenvectors for Gamma point
    q_index = 0
    eigenvectors = modes.eigenvectors[q_index]  # Shape: (num_modes, num_atoms, 3)
    masses = np.array([atom.mass for atom in modes.primitive_cell])

    print(f"\nEigenvector shape: {eigenvectors.shape}")
    print(f"Masses: {masses}")

    # Test 1: Check normalization of individual modes
    print(f"\n--- Test 1: Individual Mode Normalization ---")

    norms = []
    for mode_idx in range(eigenvectors.shape[0]):
        eigenvec = eigenvectors[mode_idx]  # Shape: (num_atoms, 3)

        # Calculate mass-weighted norm: sum_i m_i |u_i|^2
        mass_weighted_norm_sq = 0.0
        for atom_idx in range(len(masses)):
            atom_disp = eigenvec[atom_idx]  # Shape: (3,)
            mass_weighted_norm_sq += masses[atom_idx] * np.real(
                np.sum(atom_disp.conj() * atom_disp)
            )

        norm = np.sqrt(mass_weighted_norm_sq)
        norms.append(norm)

        print(
            f"Mode {mode_idx:2d} ({modes.frequencies[0][mode_idx]:8.2f} cm^-1): norm = {norm:.6f}"
        )

    print(f"All norms: {norms}")
    print(f"Expected: all should be 1.0 for orthonormal basis")
    print(f"Max deviation from 1.0: {max(abs(n - 1.0) for n in norms):.6f}")

    # Test 2: Check orthogonality between modes
    print(f"\n--- Test 2: Mode Orthogonality ---")

    num_modes = eigenvectors.shape[0]
    overlap_matrix = np.zeros((num_modes, num_modes), dtype=complex)

    for i in range(num_modes):
        for j in range(num_modes):
            eigenvec_i = eigenvectors[i]
            eigenvec_j = eigenvectors[j]

            # Calculate mass-weighted inner product: sum_k m_k u_i,k^* · u_j,k
            overlap = 0.0
            for atom_idx in range(len(masses)):
                atom_i = eigenvec_i[atom_idx]
                atom_j = eigenvec_j[atom_idx]
                overlap += masses[atom_idx] * np.sum(atom_i.conj() * atom_j)

            overlap_matrix[i, j] = overlap

    # Check if overlap matrix is identity
    identity = np.eye(num_modes)
    overlap_real = np.real(overlap_matrix)

    print(f"Overlap matrix (real part):")
    print(overlap_real)
    print(f"\nDifference from identity:")
    diff_from_identity = overlap_real - identity
    print(diff_from_identity)
    print(
        f"\nMax off-diagonal element: {np.max(np.abs(diff_from_identity - np.diag(np.diag(diff_from_identity)))):.6f}"
    )
    print(
        f"Max diagonal deviation from 1: {np.max(np.abs(np.diag(diff_from_identity))):.6f}"
    )

    # Test 3: Check if this explains our projection coefficient sum issue
    print(f"\n--- Test 3: Projection Coefficient Analysis ---")

    # Generate a displacement from mode 14
    mode_idx = 14
    supercell_matrix = np.eye(3, dtype=int)  # 1x1x1

    single_mode_disp = modes.generate_mode_displacement(
        q_index=0,
        mode_index=mode_idx,
        supercell_matrix=supercell_matrix,
        amplitude=1.0,
    )

    print(f"Generated displacement for mode {mode_idx}")
    print(f"Displacement shape: {single_mode_disp.shape}")

    # Manually calculate projection coefficients
    from phonproj.modes import create_supercell

    target_supercell = create_supercell(modes.primitive_cell, supercell_matrix)
    supercell_masses = np.array([atom.mass for atom in target_supercell])

    print(f"Supercell masses: {supercell_masses}")

    # Calculate projection onto each mode
    projection_coeffs = []
    for m_idx in range(num_modes):
        # Get mode displacement in supercell
        mode_disp_supercell = modes.generate_mode_displacement(
            q_index=0,
            mode_index=m_idx,
            supercell_matrix=supercell_matrix,
            amplitude=1.0,
        )

        # Calculate mass-weighted inner product
        numerator = 0.0
        denominator = 0.0

        for atom_idx in range(len(supercell_masses)):
            mass = supercell_masses[atom_idx]
            target_vec = single_mode_disp[atom_idx]
            mode_vec = mode_disp_supercell[atom_idx]

            numerator += mass * np.sum(target_vec.conj() * mode_vec).real
            denominator += mass * np.sum(mode_vec.conj() * mode_vec).real

        projection_coeff = numerator / denominator if denominator > 0 else 0.0
        projection_coeffs.append(projection_coeff)

        if abs(projection_coeff) > 1e-6:
            print(
                f"Mode {m_idx:2d}: coeff = {projection_coeff:.6f}, coeff^2 = {projection_coeff**2:.6f}"
            )

    sum_squared_coeffs = sum(c**2 for c in projection_coeffs)
    print(f"\nSum of squared coefficients: {sum_squared_coeffs:.6f}")
    print(f"Expected: 1.0 for orthonormal basis")
    print(f"Error: {abs(sum_squared_coeffs - 1.0):.6f}")

    # Test 4: Check if the issue is in mode generation
    print(f"\n--- Test 4: Mode Generation Consistency ---")

    # Check if generate_mode_displacement produces normalized modes
    test_mode_disp = modes.generate_mode_displacement(
        q_index=0,
        mode_index=mode_idx,
        supercell_matrix=supercell_matrix,
        amplitude=1.0,
    )

    # Calculate its mass-weighted norm
    norm_squared = 0.0
    for atom_idx in range(len(supercell_masses)):
        mass = supercell_masses[atom_idx]
        atom_vec = test_mode_disp[atom_idx]
        norm_squared += mass * np.sum(atom_vec.conj() * atom_vec).real

    norm = np.sqrt(norm_squared)
    print(f"Generated mode displacement norm: {norm:.6f}")
    print(f"Expected: depends on amplitude and normalization convention")

    return overlap_matrix, projection_coeffs, sum_squared_coeffs


if __name__ == "__main__":
    overlap_matrix, projection_coeffs, sum_squared = test_phonon_orthonormality()
