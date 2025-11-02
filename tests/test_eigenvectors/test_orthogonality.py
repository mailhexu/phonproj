#!/usr/bin/env python3

import numpy as np
from pathlib import Path
import sys

# Add the project root to the path
project_root = Path(__file__).parent.parent.parent
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

modes = PhononModes.from_phonopy_yaml(str(BATIO3_YAML_PATH), qpoints_2x2x2)
supercell_matrix = np.eye(3, dtype=int) * 2

# Generate all commensurate displacements with proper scaling
all_commensurate_displacements = modes.generate_all_commensurate_displacements(
    supercell_matrix, amplitude=1.0
)

# Flatten all modes into a single list for orthogonality checking
all_modes = []
q_mode_labels = []
supercell_masses = np.tile(modes.atomic_masses, 8)  # 2x2x2 supercell

for q_index, displacements in all_commensurate_displacements.items():
    for mode_idx in range(displacements.shape[0]):
        all_modes.append(displacements[mode_idx])
        q_mode_labels.append((q_index, mode_idx))

print(f"Total modes collected: {len(all_modes)}")
print(f"Each mode shape: {all_modes[0].shape}")

# Check orthogonality between a few modes from different q-points
print("\n=== Orthogonality Check ===")
test_pairs = [
    (0, 1),  # Same q-point, different modes
    (0, 15),  # Different q-points, mode 0
    (0, 30),  # Different q-points, mode 0
    (5, 20),  # Different q-points, different modes
]

for i, j in test_pairs:
    if i < len(all_modes) and j < len(all_modes):
        q1, m1 = q_mode_labels[i]
        q2, m2 = q_mode_labels[j]

        # Calculate mass-weighted inner product
        inner_product = modes.mass_weighted_projection(
            all_modes[i], all_modes[j], supercell_masses
        )

        print(f"Modes ({q1},{m1}) × ({q2},{m2}): inner product = {inner_product:.8f}")

# Check completeness sum again for verification
total_norm_squared = 0.0
for mode in all_modes:
    norm = modes.mass_weighted_norm(mode, supercell_masses)
    total_norm_squared += norm**2

print(f"\n=== Completeness Verification ===")
print(f"Total norm² sum: {total_norm_squared:.12f}")
print(f"Expected: 1.0")
print(f"Difference: {abs(total_norm_squared - 1.0):.12f}")
