#!/usr/bin/env python3

import numpy as np
from pathlib import Path
import sys

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

modes = PhononModes.from_phonopy_yaml(str(BATIO3_YAML_PATH), qpoints_2x2x2)
supercell_matrix = np.eye(3, dtype=int) * 2

# Find a non-Gamma q-point (not [0,0,0])
non_gamma_index = None
for i, qpoint in enumerate(modes.qpoints):
    if not np.allclose(qpoint, [0, 0, 0], atol=1e-6):
        non_gamma_index = i
        print(f"Using non-Gamma q-point {i}: {qpoint}")
        break

if non_gamma_index is None:
    print("No non-Gamma q-points found!")
    exit(1)

# Generate displacements for all modes at non-Gamma point
all_displacements = modes.generate_all_mode_displacements(
    non_gamma_index, supercell_matrix, amplitude=1.0
)

print(f"Generated {all_displacements.shape[0]} modes")
print(f"Each mode shape: {all_displacements.shape[1:]} (should be 40 atoms × 3)")

# Check individual mode norms
print("\nMode norms:")
supercell_masses = np.tile(modes.atomic_masses, 8)  # 2x2x2 supercell
for i in range(min(6, all_displacements.shape[0])):
    norm = modes.mass_weighted_norm(all_displacements[i], supercell_masses)
    print(f"  Mode {i}: norm = {norm:.8f}")

# Check problematic orthogonality between modes 0 and 1
projection_01 = modes.mass_weighted_projection(
    all_displacements[0], all_displacements[1], supercell_masses
)
print(f"\nProjection between modes 0 and 1: {projection_01}")

# Sample some displacement values to see if they look correct
print(f"\nMode 0 sample displacements (first 3 atoms):")
print(all_displacements[0][:3])
print(f"\nMode 1 sample displacements (first 3 atoms):")
print(all_displacements[1][:3])

# Check if displacements are all identical (sign of phonopy API issue)
print(
    f"\nAre all atoms in mode 0 identical? {np.allclose(all_displacements[0], all_displacements[0][0])}"
)
print(
    f"Are all atoms in mode 1 identical? {np.allclose(all_displacements[1], all_displacements[1][0])}"
)
