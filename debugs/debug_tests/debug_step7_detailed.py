#!/usr/bin/env python3

import numpy as np
import sys
from pathlib import Path

# Add the project root to the path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from phonproj.modes import PhononModes

# Load BaTiO3 data with full 2x2x2 q-point grid
BATIO3_YAML_PATH = project_root / "data" / "BaTiO3_phonopy_params.yaml"

qpoints_2x2x2 = []
for i in range(2):
    for j in range(2):
        for k in range(2):
            qpoints_2x2x2.append([i / 2.0, j / 2.0, k / 2.0])
qpoints_2x2x2 = np.array(qpoints_2x2x2)

print("Q-points:")
for i, qpt in enumerate(qpoints_2x2x2):
    print(f"  {i}: {qpt}")

modes = PhononModes.from_phonopy_yaml(str(BATIO3_YAML_PATH), qpoints_2x2x2)
supercell_matrix = np.eye(3, dtype=int) * 2

print(f"\nEigenvectors shape: {modes.eigenvectors.shape}")
print(f"Frequencies shape: {modes.frequencies.shape}")

# Generate all commensurate displacements and analyze each q-point
all_commensurate_displacements = modes.generate_all_commensurate_displacements(
    supercell_matrix, amplitude=1.0
)

print(f"\nFound {len(all_commensurate_displacements)} commensurate q-points")

total_modes = 0
total_norm_squared = 0

for q_index, displacements in all_commensurate_displacements.items():
    qpoint = modes.qpoints[q_index]
    n_modes = displacements.shape[0]
    print(f"\nQ-point {q_index} {qpoint}: {n_modes} modes")

    # Check individual mode norms
    supercell_masses = np.tile(modes.atomic_masses, 8)  # 2x2x2 supercell

    for mode_idx in range(n_modes):  # Sum ALL modes, not just first 6
        norm = modes.mass_weighted_norm(displacements[mode_idx], supercell_masses)
        norm_squared = norm**2
        total_norm_squared += norm_squared

        if mode_idx < 6:  # Only print first 6 for readability
            print(f"  Mode {mode_idx}: norm = {norm:.8f}, norm² = {norm_squared:.8f}")

        if norm < 1e-10:
            print(f"    WARNING: Near-zero norm for mode {mode_idx}")

    total_modes += n_modes

print(f"\nSummary:")
print(f"Total modes: {total_modes}")
print(f"Total norm² sum: {total_norm_squared:.8f}")
print(f"Expected: 1.0 (completeness)")
print(f"Difference: {abs(total_norm_squared - 1.0):.8f}")

# Also check for a specific q-point that was problematic before
print(f"\n=== Detailed Analysis of Edge Q-point [0, 0, 0.5] ===")

# Find q-point [0, 0, 0.5]
edge_q_index = None
for i, qpt in enumerate(modes.qpoints):
    if np.allclose(qpt, [0, 0, 0.5], atol=1e-6):
        edge_q_index = i
        break

if edge_q_index is not None:
    print(f"Found edge q-point at index {edge_q_index}")

    # Generate displacements for this specific q-point
    edge_displacements = modes.generate_all_mode_displacements(
        edge_q_index, supercell_matrix, amplitude=1.0
    )

    supercell_masses = np.tile(modes.atomic_masses, 8)

    print(f"Generated {edge_displacements.shape[0]} modes for edge q-point")

    for mode_idx in range(min(6, edge_displacements.shape[0])):
        norm = modes.mass_weighted_norm(edge_displacements[mode_idx], supercell_masses)
        print(f"Mode {mode_idx}: norm = {norm:.12f}")

        # Show some displacement values
        disp_sample = edge_displacements[mode_idx][:3]  # First 3 components
        print(f"  Sample displacements: {disp_sample}")
else:
    print("Edge q-point [0, 0, 0.5] not found")
