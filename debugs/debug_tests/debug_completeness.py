#!/usr/bin/env python3

import numpy as np
from pathlib import Path
from phonproj.modes import PhononModes


def debug_completeness():
    """Debug the completeness test to understand the normalization issue."""

    # Test data path
    BATIO3_YAML_PATH = Path("data/BaTiO3_phonopy_params.yaml")

    # Load BaTiO3 data for 2x2x2 supercell
    qpoints_2x2x2 = []
    for i in range(2):
        for j in range(2):
            for k in range(2):
                qpoints_2x2x2.append([i / 2.0, j / 2.0, k / 2.0])
    qpoints_2x2x2 = np.array(qpoints_2x2x2)

    modes = PhononModes.from_phonopy_yaml(str(BATIO3_YAML_PATH), qpoints_2x2x2)

    # Use 2x2x2 supercell
    supercell_matrix = np.eye(3, dtype=int) * 2

    print("=== ANALYZING EIGENMODE NORMALIZATION ===")
    print(f"Supercell matrix: 2x2x2, N = 8 primitive cells")
    print(f"Expected norm per eigenmode (theory): 1/√N = 1/√8 = {1.0 / np.sqrt(8):.6f}")

    # Generate displacements for all commensurate q-points
    all_commensurate_displacements = modes.generate_all_commensurate_displacements(
        supercell_matrix, amplitude=1.0
    )

    total_modes = 0
    sum_norms_squared = 0.0

    print("\n=== CURRENT NORMALIZATION STATUS ===")
    for q_index, displacements in all_commensurate_displacements.items():
        print(f"\nQ-point {q_index}: {len(displacements)} modes")
        for i in range(min(3, len(displacements))):  # Show first 3 modes
            norm = modes.mass_weighted_norm(displacements[i])
            print(f"  Mode {i}: norm = {norm:.6f}")
            sum_norms_squared += norm**2
            total_modes += 1

    print(f"\nTotal modes analyzed: {total_modes}")
    print(f"Sum of norms squared: {sum_norms_squared:.6f}")
    print(f"Average norm: {np.sqrt(sum_norms_squared / total_modes):.6f}")

    # Test completeness with current normalization
    print("\n=== COMPLETENESS TEST WITH CURRENT NORMS ===")

    # Create random displacement
    np.random.seed(123)
    n_supercell_atoms = 8 * modes._n_atoms  # 2x2x2 supercell
    random_displacement = np.random.rand(n_supercell_atoms, 3)

    # Normalize with mass-weighted norm 1
    supercell_masses = np.tile(modes.atomic_masses, 8)
    current_norm = modes.mass_weighted_norm(random_displacement, supercell_masses)
    normalized_displacement = random_displacement / current_norm

    # Project onto all eigendisplacements
    sum_projections_squared = 0.0

    for q_index, displacements in all_commensurate_displacements.items():
        for i in range(displacements.shape[0]):
            projection = modes.mass_weighted_projection(
                normalized_displacement, displacements[i], supercell_masses
            )
            sum_projections_squared += abs(projection) ** 2

    print(f"Sum of projections squared: {sum_projections_squared:.6f}")
    print(f"Theoretical sum (perfect orthonormal): 1.0")
    print(f"Ratio: {sum_projections_squared:.2f}x")

    # Test with corrected normalization
    print("\n=== TESTING WITH CORRECTED NORMALIZATION ===")

    # Re-normalize each eigenmode to have norm 1/√N
    N = 8  # 2x2x2 supercell
    target_norm = 1.0 / np.sqrt(N)

    corrected_displacements = {}

    for q_index, displacements in all_commensurate_displacements.items():
        corrected_modes = []
        for i in range(displacements.shape[0]):
            current_mode_norm = modes.mass_weighted_norm(displacements[i])
            if current_mode_norm > 1e-12:
                corrected_mode = displacements[i] * target_norm / current_mode_norm
                corrected_modes.append(corrected_mode)
        corrected_displacements[q_index] = np.array(corrected_modes)

    # Test completeness with corrected normalization
    sum_projections_squared_corrected = 0.0

    for q_index, displacements in corrected_displacements.items():
        for i in range(displacements.shape[0]):
            projection = modes.mass_weighted_projection(
                normalized_displacement, displacements[i], supercell_masses
            )
            sum_projections_squared_corrected += abs(projection) ** 2

    print(f"Sum with corrected norms: {sum_projections_squared_corrected:.6f}")
    print(f"Ratio with corrected norms: {sum_projections_squared_corrected:.2f}x")

    # Verify corrected normalization
    print(f"\nVerifying corrected normalization:")
    first_q = list(corrected_displacements.keys())[0]
    first_mode_norm = modes.mass_weighted_norm(corrected_displacements[first_q][0])
    print(f"First corrected mode norm: {first_mode_norm:.6f}")
    print(f"Target norm: {target_norm:.6f}")


if __name__ == "__main__":
    debug_completeness()
