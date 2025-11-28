"""
Debug script for uniform displacement projection bug.

PURPOSE:
    Investigate why uniform displacement incorrectly projects onto optical modes
    instead of only Gamma acoustic modes. The issue appears to be inconsistent
    mass-weighting between different supercell cases.

USAGE:
    uv run python agent_files/debug_uniform_projection_bug/debug_main.py

EXPECTED OUTPUT:
    Analysis of mass-weighting consistency and projection behavior
    for uniform displacement in different supercell configurations.

FILES USED:
    - phonproj/modes.py: PhononModes class and displacement generation
    - tests/test_displacement/test_uniform_projection.py: Failing test
    - data/BaTiO3_phonopy_params.yaml: Test data

DEBUG NOTES:
    The test fails because uniform displacement projects onto optical modes (0.882)
    instead of only Gamma acoustic modes. This suggests a bug in the mass-weighting
    or normalization of mode displacements.
"""

import numpy as np
from pathlib import Path

from phonproj.modes import PhononModes


def main():
    print("=== Debugging Uniform Displacement Projection Bug ===\n")

    # Load BaTiO3 data
    data_path = (
        Path(__file__).parent.parent.parent / "data" / "BaTiO3_phonopy_params.yaml"
    )
    modes_gamma = PhononModes.from_phonopy_yaml(
        str(data_path), qpoints=np.array([[0.0, 0.0, 0.0]])
    )

    print(
        f"Loaded BaTiO3 modes at Gamma: {modes_gamma.n_modes} modes, {modes_gamma.n_atoms} atoms\n"
    )

    # Test 1: Check mass-weighting consistency for Gamma point unit cell
    print("Test 1: Gamma point unit cell mass-weighting")
    print("-" * 50)

    supercell_matrix_unit = np.eye(3)
    all_displacements_unit = modes_gamma.generate_all_mode_displacements(
        q_index=0, supercell_matrix=supercell_matrix_unit, amplitude=1.0
    )

    # Check orthonormality under mass-weighted inner product
    n_modes = all_displacements_unit.shape[0]
    orthonormality_matrix = np.zeros((n_modes, n_modes), dtype=complex)

    supercell_masses = np.tile(modes_gamma.atomic_masses, 1)  # Unit cell

    for i in range(n_modes):
        for j in range(n_modes):
            disp_i = all_displacements_unit[i].flatten()
            disp_j = all_displacements_unit[j].flatten()
            projection = modes_gamma.mass_weighted_projection(
                disp_i, disp_j, supercell_masses
            )
            orthonormality_matrix[i, j] = projection

    max_error = np.max(np.abs(orthonormality_matrix - np.eye(n_modes)))
    print(f"Max orthonormality error: {max_error:.6f}")
    print(f"Diagonal elements: {np.diag(orthonormality_matrix.real)}")
    print(
        f"Max off-diagonal: {np.max(np.abs(orthonormality_matrix - np.eye(n_modes)))}"
    )

    # Test 2: Check mass-weighting for 4x1x1 supercell
    print("\nTest 2: 4x1x1 supercell mass-weighting")
    print("-" * 50)

    supercell_matrix_4x1x1 = np.diag([4, 1, 1])
    all_displacements_4x1x1 = modes_gamma.generate_all_mode_displacements(
        q_index=0, supercell_matrix=supercell_matrix_4x1x1, amplitude=1.0
    )

    # Check orthonormality
    n_supercell_atoms = all_displacements_4x1x1.shape[1]
    supercell_masses_4x1x1 = np.tile(modes_gamma.atomic_masses, 4)

    orthonormality_matrix_4x1x1 = np.zeros((n_modes, n_modes), dtype=complex)

    for i in range(n_modes):
        for j in range(n_modes):
            disp_i = all_displacements_4x1x1[i].flatten()
            disp_j = all_displacements_4x1x1[j].flatten()
            projection = modes_gamma.mass_weighted_projection(
                disp_i, disp_j, supercell_masses_4x1x1
            )
            orthonormality_matrix_4x1x1[i, j] = projection

    max_error_4x1x1 = np.max(np.abs(orthonormality_matrix_4x1x1 - np.eye(n_modes)))
    print(".6f")
    print(f"Diagonal elements: {np.diag(orthonormality_matrix_4x1x1.real)}")
    print(
        f"Max off-diagonal: {np.max(np.abs(orthonormality_matrix_4x1x1 - np.eye(n_modes)))}"
    )

    # Test 3: Check projection of uniform displacement
    print("\nTest 3: Uniform displacement projection analysis")
    print("-" * 50)

    # Create uniform displacement for 4x1x1 supercell
    uniform_disp = np.ones((n_supercell_atoms, 3))
    uniform_norm = modes_gamma.mass_weighted_norm(uniform_disp, supercell_masses_4x1x1)
    uniform_disp = uniform_disp / uniform_norm

    print(f"Uniform displacement norm: {uniform_norm:.6f}")
    print(
        f"Normalized uniform displacement norm: {modes_gamma.mass_weighted_norm(uniform_disp, supercell_masses_4x1x1):.6f}"
    )

    # Project onto all modes
    projections = []
    for mode_idx in range(n_modes):
        mode_disp = all_displacements_4x1x1[mode_idx].flatten()
        coeff = modes_gamma.mass_weighted_projection_coefficient(
            mode_disp, uniform_disp.flatten(), supercell_masses_4x1x1, debug=False
        )
        projections.append(abs(coeff) ** 2)

        freq = modes_gamma.frequencies[0, mode_idx]
        if abs(coeff) > 1e-6:
            mode_type = "acoustic" if mode_idx < 3 else "optical"
            print(
                f"Mode {mode_idx}: freq = {freq:8.2f} THz, |coeff|² = {abs(coeff)**2:.6f} ({mode_type})"
            )
    projections = np.array(projections)
    total_projection = np.sum(projections)

    print(f"\nTotal projection sum: {total_projection:.6f}")
    print(f"Gamma acoustic projection (modes 0-2): {np.sum(projections[:3]):.6f}")
    print(f"Optical projection (modes 3+): {np.sum(projections[3:]):.6f}")

    # Test 4: Compare with raw eigenvector projections
    print("\nTest 4: Raw eigenvector analysis")
    print("-" * 50)

    # For Gamma point, eigenvectors should be real
    eigenvectors = modes_gamma.eigenvectors[0]  # Shape: (n_modes, n_atoms*3)

    # Check if eigenvectors are orthonormal under plain inner product
    plain_orthonormality = np.zeros((n_modes, n_modes), dtype=complex)
    for i in range(n_modes):
        for j in range(n_modes):
            plain_orthonormality[i, j] = np.vdot(eigenvectors[i], eigenvectors[j])

    plain_max_error = np.max(np.abs(plain_orthonormality - np.eye(n_modes)))
    print(f"Raw eigenvector orthonormality error: {plain_max_error:.6f}")

    # Project uniform displacement onto raw eigenvectors
    uniform_flat = uniform_disp.flatten()
    # For primitive cell projection, we need to project onto the primitive eigenvectors
    # But uniform displacement is defined on supercell, so this is tricky

    print("\n=== Analysis Complete ===")


if __name__ == "__main__":
    main()
