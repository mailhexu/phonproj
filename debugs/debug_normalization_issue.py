#!/usr/bin/env python3
"""
Debug the normalization issue in mode decomposition.
We expect projection of a mode onto itself to equal 1.0, but getting ~1.04.
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
print("DEBUG: Normalization Issue in Mode Decomposition")
print("=" * 80)
print(f"\nDataset: BaTiO3")
print(f"Supercell: 1x1x1")
print(f"Q-points: {len(modes.qpoints)} (Gamma only)")
print(f"Modes: {modes.frequencies.shape[1]}")

# Generate a mode displacement (q=0, mode=14)
q_index = 0
mode_index = 14

print(f"\n--- Generating Mode Displacement ---")
print(f"Q-point index: {q_index}")
print(f"Mode index: {mode_index}")

mode_disp = modes.generate_mode_displacement(
    q_index=q_index,
    mode_index=mode_index,
    supercell_matrix=supercell_matrix,
    amplitude=1.0,
)

print(f"Displacement shape: {mode_disp.shape}")

# Calculate mass-weighted norm
supercell = modes.generate_supercell(supercell_matrix)
masses = supercell.get_masses()
masses_repeated = np.repeat(masses, 3)
mode_flat = mode_disp.ravel()

# Norm BEFORE any normalization
raw_norm_sq = np.sum(masses_repeated * np.abs(mode_flat) ** 2)
raw_norm = np.sqrt(raw_norm_sq)

print(f"\n--- Mass-Weighted Norm (As Generated) ---")
print(f"Norm²: {raw_norm_sq:.10f}")
print(f"Norm:  {raw_norm:.10f}")

# This is what decompose_displacement_to_modes does to normalize the mode
normalized_mode_disp = mode_disp / raw_norm
normalized_flat = normalized_mode_disp.ravel()
normalized_norm_sq = np.sum(masses_repeated * np.abs(normalized_flat) ** 2)

print(f"\n--- After Normalization (divide by raw_norm) ---")
print(f"Norm²: {normalized_norm_sq:.10f}")
print(f"Norm:  {np.sqrt(normalized_norm_sq):.10f}")

# Now decompose this mode displacement back onto itself
print(f"\n--- Decomposing Mode Onto Itself ---")
results, summary = decompose_displacement_to_modes(
    mode_disp, modes, supercell_matrix, normalize=False
)

print(f"Sum of squared projections: {summary['sum_squared_projections']:.10f}")
print(f"Expected: 1.000000")
print(f"Error: {abs(summary['sum_squared_projections'] - 1.0):.10f}")

# Find the dominant mode
results_sorted = sorted(results, key=lambda x: x["squared_coefficient"], reverse=True)
print(f"\nTop 3 contributions:")
for i, r in enumerate(results_sorted[:3]):
    print(
        f"  {i + 1}. Q={r['q_index']}, Mode={r['mode_index']}: "
        f"coeff²={r['squared_coefficient']:.10f}"
    )

# The key insight: Check what happens inside decompose_displacement_to_modes
print(f"\n--- Inside decompose_displacement_to_modes ---")
print("The function does:")
print("1. Generate mode displacement (includes 1/√n_cells)")
print("2. Normalize by mass-weighted norm")
print("3. Project onto target displacement")
print()
print("The issue is that we're normalizing twice:")
print("  - Once with 1/√n_cells in generate_mode_displacement")
print("  - Again in decompose_displacement_to_modes")
print()
print("For 1x1x1: n_cells=1, so 1/√n_cells = 1.0 (no scaling)")
print("But eigenvectors might already have their own normalization...")

# Check eigenvector norm
eigvec = modes.eigenvectors[q_index, mode_index]
eigvec_flat = eigvec.ravel()
eigvec_norm_sq = np.sum(masses_repeated[: len(eigvec_flat)] * np.abs(eigvec_flat) ** 2)
eigvec_norm = np.sqrt(eigvec_norm_sq)

print(f"\n--- Eigenvector Normalization ---")
print(f"Eigenvector shape: {eigvec.shape}")
print(f"Mass-weighted norm²: {eigvec_norm_sq:.10f}")
print(f"Mass-weighted norm:  {eigvec_norm:.10f}")
print()
print("If eigenvector mass-weighted norm = 1.0, it's already normalized")
print("If it's not 1.0, that might be causing issues...")

print("\n" + "=" * 80)
print("CONCLUSION:")
if abs(eigvec_norm - 1.0) < 1e-6:
    print("✓ Eigenvectors ARE mass-weighted normalized to 1.0")
else:
    print(f"✗ Eigenvectors have norm = {eigvec_norm:.6f}, not 1.0!")
    print("  This inconsistent normalization causes the completeness issue.")

print("=" * 80)
