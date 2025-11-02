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

modes = PhononModes.from_phonopy_yaml(str(BATIO3_YAML_PATH), qpoints_2x2x2)
supercell_matrix = np.eye(3, dtype=int) * 2

print("=== Theoretical Analysis ===")
print(f"Primitive cell atoms: {modes._n_atoms}")
print(f"Supercell atoms: {8 * modes._n_atoms}")
print(f"Modes per q-point: {15}")
print(f"Total q-points: {8}")
print(f"Total modes: {8 * 15}")

print(f"\nExpected individual mode norm: 1/sqrt(N) = 1/sqrt(8) = {1 / np.sqrt(8):.8f}")
print(f"Expected individual mode norm²: 1/N = 1/8 = {1 / 8:.8f}")
print(f"Expected total norm² for completeness: {8 * 15 * (1 / 8):.8f}")

print("\nWait, this suggests the issue!")
print("If each mode has norm² = 1/8 and we have 120 modes,")
print("then total norm² = 120 * (1/8) = 15, not 1!")
print("This means the individual mode normalization is wrong.")

print("\n=== Current Implementation Analysis ===")

# Generate a single displacement to check normalization
q_idx = 0  # Gamma point
mode_idx = 3  # Some mode

displacement = modes.generate_all_mode_displacements(
    q_idx, supercell_matrix, amplitude=1.0
)[mode_idx]
supercell_masses = np.tile(modes.atomic_masses, 8)

actual_norm = modes.mass_weighted_norm(displacement, supercell_masses)
print(f"Current mode norm: {actual_norm:.8f}")
print(f"Current mode norm²: {actual_norm**2:.8f}")

print(f"\n=== Checking Theory ===")
print("For completeness, we need:")
print("Sum of |<ψ_random | ψ_i>|² = 1")
print("If modes are orthonormal: <ψ_i | ψ_j> = δ_ij")
print("Then each mode should have mass-weighted norm = 1")
print("NOT norm = 1/sqrt(N)")

print(f"\n=== Testing Orthonormality ===")
# Check if two different modes are orthogonal
displacement1 = modes.generate_all_mode_displacements(
    0, supercell_matrix, amplitude=1.0
)[0]  # mode 0
displacement2 = modes.generate_all_mode_displacements(
    0, supercell_matrix, amplitude=1.0
)[1]  # mode 1

projection = modes.mass_weighted_projection(
    displacement1, displacement2, supercell_masses
)
print(f"Projection between modes 0 and 1: {projection:.8f}")
print(f"Should be ~0 for orthogonal modes")

norm1 = modes.mass_weighted_norm(displacement1, supercell_masses)
norm2 = modes.mass_weighted_norm(displacement2, supercell_masses)
print(f"Mode 0 norm: {norm1:.8f}")
print(f"Mode 1 norm: {norm2:.8f}")
print(f"For orthonormal basis, these should be 1.0")

print(f"\n=== The Fix ===")
print("The issue is likely in the amplitude scaling!")
print("We should normalize each mode to have mass-weighted norm = 1")
print("Current amplitude=1.0 gives norm = 0.125")
print(f"So we should use amplitude = {1.0 / actual_norm:.8f}")

# Test the fix
fixed_amplitude = 1.0 / actual_norm
displacement_fixed = modes.generate_all_mode_displacements(
    q_idx, supercell_matrix, amplitude=fixed_amplitude
)[mode_idx]
fixed_norm = modes.mass_weighted_norm(displacement_fixed, supercell_masses)
print(f"Fixed mode norm: {fixed_norm:.8f}")
print(f"This should be 1.0!")
