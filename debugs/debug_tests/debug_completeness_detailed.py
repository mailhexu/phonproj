#!/usr/bin/env python3

import numpy as np
import sys

sys.path.append("/Users/hexu/projects/phonproj")

from phonproj.modes import PhononModes

# Set up the same test data as in the test
BATIO3_YAML_PATH = "/Users/hexu/projects/phonproj/data/BaTiO3_phonopy_params.yaml"

# Load BaTiO3 data - need full 2x2x2 q-point grid
qpoints_2x2x2 = []
for i in range(2):
    for j in range(2):
        for k in range(2):
            qpoints_2x2x2.append([i / 2.0, j / 2.0, k / 2.0])
qpoints_2x2x2 = np.array(qpoints_2x2x2)

modes = PhononModes.from_phonopy_yaml(str(BATIO3_YAML_PATH), qpoints_2x2x2)

# Use 2x2x2 supercell
supercell_matrix = np.eye(3, dtype=int) * 2

# Generate displacements for all commensurate q-points
all_commensurate_displacements = modes.generate_all_commensurate_displacements(
    supercell_matrix, amplitude=1.0
)

print("=== 2x2x2 Completeness Projection Debug ===")

# Create a random displacement for supercell (same as test)
np.random.seed(123)  # Same seed as test
n_supercell_atoms = 8 * modes._n_atoms  # 2x2x2 supercell
random_displacement = np.random.rand(n_supercell_atoms, 3)

# Normalize with mass-weighted norm 1
supercell_masses = np.tile(modes.atomic_masses, 8)  # Repeat for each primitive cell
current_norm = modes.mass_weighted_norm(random_displacement, supercell_masses)
normalized_displacement = random_displacement / current_norm

print(f"Random displacement shape: {random_displacement.shape}")
print(f"Supercell masses shape: {supercell_masses.shape}")
print(f"Original norm: {current_norm}")

# Verify normalization
check_norm = modes.mass_weighted_norm(normalized_displacement, supercell_masses)
print(f"Normalized displacement norm: {check_norm}")

# Project onto all eigendisplacements
print(f"\nProjecting onto all eigendisplacements:")
sum_projections_squared = 0.0
max_projection = 0.0
min_projection = float("inf")
projection_count = 0

# Also check individual displacement norms
print(f"\nEigen-displacement norms:")
for q_index, displacements in all_commensurate_displacements.items():
    for i in range(displacements.shape[0]):
        disp_norm = modes.mass_weighted_norm(displacements[i], supercell_masses)
        if i < 3:  # Print only first few for brevity
            print(f"  q{q_index}, mode{i}: norm = {disp_norm}")

        projection = modes.mass_weighted_projection(
            normalized_displacement, displacements[i], supercell_masses
        )
        projection_sq = abs(projection) ** 2
        sum_projections_squared += projection_sq
        projection_count += 1

        max_projection = max(max_projection, abs(projection))
        min_projection = min(min_projection, abs(projection))

        if i < 3:  # Print only first few for brevity
            print(f"    projection = {projection}, projection² = {projection_sq}")

print(f"\nProjection statistics:")
print(f"  Total projections: {projection_count}")
print(f"  Max |projection|: {max_projection}")
print(f"  Min |projection|: {min_projection}")
print(f"  Sum of projections²: {sum_projections_squared}")
print(f"  Expected sum: 1.0")
print(f"  Ratio (actual/expected): {sum_projections_squared}")

# Check if the displacement norms are correct (should be 1/N = 1/8 = 0.125)
expected_norm = 1.0 / 8
print(f"\nExpected displacement norm: {expected_norm}")

# Sample a few displacement norms
sample_norms = []
for q_index, displacements in all_commensurate_displacements.items():
    for i in range(
        min(3, displacements.shape[0])
    ):  # Sample first 3 modes from each q-point
        norm = modes.mass_weighted_norm(displacements[i], supercell_masses)
        sample_norms.append(norm)

print(f"Sample displacement norms: {sample_norms[:10]}")  # Show first 10
print(
    f"All norms ≈ {expected_norm}? {all(abs(n - expected_norm) < 1e-6 for n in sample_norms[:10])}"
)

if sum_projections_squared < 0.5:
    print(f"\nPROBLEM IDENTIFIED: Severely incomplete projection sum!")
    print(
        f"This suggests the eigendisplacements don't span the supercell space correctly."
    )
    print(f"Possible causes:")
    print(f"  1. Incorrect displacement calculation in supercell")
    print(f"  2. Wrong mass array for supercell projections")
    print(f"  3. Phase relationships in supercell construction")
