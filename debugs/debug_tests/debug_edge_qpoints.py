#!/usr/bin/env python3

import numpy as np
from pathlib import Path
from phonproj.modes import PhononModes

# Load BaTiO3 data with 2x2x2 q-point grid
qpoints_2x2x2 = []
for i in range(2):
    for j in range(2):
        for k in range(2):
            qpoints_2x2x2.append([i / 2.0, j / 2.0, k / 2.0])
qpoints_2x2x2 = np.array(qpoints_2x2x2)

print("Q-points:")
for i, q in enumerate(qpoints_2x2x2):
    print(f"  Q-point {i}: {q}")

BATIO3_YAML_PATH = Path("data/BaTiO3_phonopy_params.yaml")
modes = PhononModes.from_phonopy_yaml(str(BATIO3_YAML_PATH), qpoints_2x2x2)

# Use 2x2x2 supercell
supercell_matrix = np.eye(3, dtype=int) * 2

print("\nInvestigating each q-point:")
print("=" * 50)

for q_idx in range(len(qpoints_2x2x2)):
    qpoint = qpoints_2x2x2[q_idx]
    print(f"\nQ-point {q_idx}: {qpoint}")

    # Check if eigenvectors exist and are valid
    if q_idx < len(modes.eigenvectors):
        eigenvectors = modes.eigenvectors[q_idx]
        print(f"  Eigenvectors shape: {eigenvectors.shape}")
        print(
            f"  Eigenvector norms: {[np.linalg.norm(eigenvectors[:, i]) for i in range(min(3, eigenvectors.shape[1]))]}"
        )

        # Check if any eigenvectors are near-zero
        for mode_idx in range(eigenvectors.shape[1]):
            eigenvector = eigenvectors[:, mode_idx]
            norm = np.linalg.norm(eigenvector)
            if norm < 1e-10:
                print(
                    f"  WARNING: Mode {mode_idx} has near-zero eigenvector (norm={norm})"
                )
    else:
        print(f"  ERROR: No eigenvectors found for q-point {q_idx}")
        continue

    # Generate displacements for first few modes
    try:
        all_displacements = modes.generate_all_mode_displacements(
            q_idx, supercell_matrix, amplitude=1.0
        )

        print(f"  Generated {all_displacements.shape[0]} modes")

        # Check displacement norms
        for mode_idx in range(min(3, all_displacements.shape[0])):
            displacement = all_displacements[mode_idx]
            norm = modes.mass_weighted_norm(displacement)
            print(f"  Mode {mode_idx} mass-weighted norm: {norm:.6e}")

            # Check if displacement is essentially zero
            if norm < 1e-10:
                print(f"    WARNING: Mode {mode_idx} produces near-zero displacement!")

                # Investigate why - check intermediate values
                eigenvector = modes.eigenvectors[q_idx][:, mode_idx]
                print(f"    Eigenvector norm: {np.linalg.norm(eigenvector):.6e}")

                # Check a few atoms' displacements
                for atom_idx in range(min(3, displacement.shape[0])):
                    atom_disp = displacement[atom_idx]
                    atom_disp_norm = np.linalg.norm(atom_disp)
                    print(
                        f"    Atom {atom_idx} displacement norm: {atom_disp_norm:.6e}"
                    )
                    if atom_disp_norm > 1e-12:
                        print(f"      Displacement: {atom_disp}")

    except Exception as e:
        print(f"  ERROR generating displacements: {e}")

print("\n" + "=" * 50)
print("Summary Analysis:")

# Test completeness
print("\nTesting completeness...")
all_commensurate_displacements = modes.generate_all_commensurate_displacements(
    supercell_matrix, amplitude=1.0
)

print(f"Found {len(all_commensurate_displacements)} commensurate q-points")
total_modes = sum(
    displacements.shape[0] for displacements in all_commensurate_displacements.values()
)
print(f"Total modes: {total_modes}")

# Create test displacement
np.random.seed(123)
n_supercell_atoms = 8 * modes._n_atoms
random_displacement = np.random.rand(n_supercell_atoms, 3)

# Normalize
supercell_masses = np.tile(modes.atomic_masses, 8)
current_norm = modes.mass_weighted_norm(random_displacement, supercell_masses)
normalized_displacement = random_displacement / current_norm

# Project onto all modes
sum_projections_squared = 0.0
mode_contributions = {}

for q_index, displacements in all_commensurate_displacements.items():
    q_contribution = 0.0
    for i in range(displacements.shape[0]):
        projection = modes.mass_weighted_projection(
            normalized_displacement, displacements[i], supercell_masses
        )
        projection_sq = abs(projection) ** 2
        sum_projections_squared += projection_sq
        q_contribution += projection_sq

    mode_contributions[q_index] = q_contribution
    qpoint = qpoints_2x2x2[q_index]
    print(f"Q-point {q_index} {qpoint}: contribution = {q_contribution:.6f}")

print(f"\nTotal completeness: {sum_projections_squared:.6f}")
print(f"Missing: {1.0 - sum_projections_squared:.6f}")
