#!/usr/bin/env python3
"""
Debug part 2: Check if the issue is that input displacement needs same normalization as modes.
"""

import numpy as np
from phonproj.modes import PhononModes
from phonproj.core.structure_analysis import decompose_displacement_to_modes

# Load BaTiO3 data
gamma_qpoint = np.array([[0.0, 0.0, 0.0]])
modes = PhononModes.from_phonopy_yaml("data/BaTiO3_phonopy_params.yaml", gamma_qpoint)

# Use 1x1x1 supercell (simplest case)
supercell_matrix = np.eye(3, dtype=int)

print("=" * 80)
print("DEBUG PART 2: Input Displacement Normalization")
print("=" * 80)

# Generate a mode displacement (q=0, mode=14)
q_index = 0
mode_index = 14

mode_disp = modes.generate_mode_displacement(
    q_index=q_index,
    mode_index=mode_index,
    supercell_matrix=supercell_matrix,
    amplitude=1.0,
)

supercell = modes.generate_supercell(supercell_matrix)
masses = supercell.get_masses()
masses_repeated = np.repeat(masses, 3)

# Calculate norm of mode displacement AS GENERATED
mode_flat = mode_disp.ravel()
mode_norm_sq = np.sum(masses_repeated * np.abs(mode_flat) ** 2)
mode_norm = np.sqrt(mode_norm_sq)

print(f"\nMode displacement AS GENERATED:")
print(f"  Mass-weighted norm: {mode_norm:.6f}")

# Now try decomposing WITHOUT normalize=False (let the function normalize)
print(f"\n--- Test 1: Pass mode_disp with normalize=False ---")
results1, summary1 = decompose_displacement_to_modes(
    mode_disp, modes, supercell_matrix, normalize=False
)
print(f"  Sum of squared projections: {summary1['sum_squared_projections']:.6f}")
print(f"  Dominant contribution: {results1[0]['squared_coefficient']:.6f}")

# Now try normalizing the INPUT to unit mass-weighted norm first
print(f"\n--- Test 2: Normalize INPUT to unit mass-weighted norm first ---")
normalized_input = mode_disp / mode_norm
normalized_flat = normalized_input.ravel()
input_norm_sq = np.sum(masses_repeated * np.abs(normalized_flat) ** 2)
print(f"  Input norm after normalization: {np.sqrt(input_norm_sq):.10f}")

results2, summary2 = decompose_displacement_to_modes(
    normalized_input, modes, supercell_matrix, normalize=False
)
print(f"  Sum of squared projections: {summary2['sum_squared_projections']:.6f}")
print(f"  Dominant contribution: {results2[0]['squared_coefficient']:.6f}")

# Try with normalize=True
print(f"\n--- Test 3: Pass with normalize=True ---")
results3, summary3 = decompose_displacement_to_modes(
    mode_disp, modes, supercell_matrix, normalize=True
)
print(f"  Sum of squared projections: {summary3['sum_squared_projections']:.6f}")
print(f"  Dominant contribution: {results3[0]['squared_coefficient']:.6f}")

print("\n" + "=" * 80)
print("ANALYSIS:")
print("The key insight is that decompose_displacement_to_modes normalizes")
print("the MODE displacements to unit mass-weighted norm, but the INPUT")
print("displacement needs to have the SAME normalization for inner products")
print("to be computed correctly.")
print()
print("Options:")
print("1. Both normalized to unit mass-weighted norm → sum should be 1.0")
print("2. Both unnormalized (raw) → sum can be anything")
print("3. Mixed → inconsistent results")
print("=" * 80)
