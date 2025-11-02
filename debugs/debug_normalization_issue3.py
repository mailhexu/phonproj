#!/usr/bin/env python3
"""
Debug part 3: Check if the 4% error is due to incorrect mass-weighted inner product.
"""

import numpy as np
from phonproj.modes import PhononModes

# Load BaTiO3 data
gamma_qpoint = np.array([[0.0, 0.0, 0.0]])
modes = PhononModes.from_phonopy_yaml("data/BaTiO3_phonopy_params.yaml", gamma_qpoint)

# Use 1x1x1 supercell (simplest case)
supercell_matrix = np.eye(3, dtype=int)
supercell = modes.generate_supercell(supercell_matrix)

print("=" * 80)
print("DEBUG PART 3: Mass-Weighted Inner Product Check")
print("=" * 80)

# Generate mode displacement for mode 14
q_index = 0
mode_index = 14

mode_disp = modes.generate_mode_displacement(
    q_index=q_index,
    mode_index=mode_index,
    supercell_matrix=supercell_matrix,
    amplitude=1.0,
)

# Get masses
masses = supercell.get_masses()
masses_repeated = np.repeat(masses, 3)

# Normalize to unit mass-weighted norm
mode_flat = mode_disp.ravel()
mode_norm = np.sqrt(np.sum(masses_repeated * np.abs(mode_flat) ** 2))
normalized_mode = mode_disp / mode_norm
normalized_flat = normalized_mode.ravel()

print(f"\nMode {mode_index} normalized to unit mass-weighted norm")
print(
    f"Check norm: {np.sqrt(np.sum(masses_repeated * np.abs(normalized_flat) ** 2)):.10f}"
)

# Manual inner product: <mode | mode> should be 1.0
inner_product = np.sum(masses_repeated * normalized_flat.conj() * normalized_flat)
print(f"\nManual <mode|mode> calculation:")
print(f"  <mode|mode> = {inner_product.real:.10f}")
print(f"  Expected: 1.0")
print(f"  Error: {abs(inner_product.real - 1.0):.10e}")

# Now check all modes - are they orthonormal?
print(f"\n--- Checking ALL modes for orthonormality ---")
n_modes = modes.n_modes

# Generate all mode displacements and normalize them
all_modes_normalized = []
for m in range(n_modes):
    md = modes.generate_mode_displacement(
        q_index=q_index,
        mode_index=m,
        supercell_matrix=supercell_matrix,
        amplitude=1.0,
    )
    md_flat = md.ravel()
    md_norm = np.sqrt(np.sum(masses_repeated * np.abs(md_flat) ** 2))
    md_normalized = md / md_norm
    all_modes_normalized.append(md_normalized.ravel())

# Check inner products between all pairs
print(f"\nGram matrix (should be identity):")
print("     ", end="")
for j in range(min(n_modes, 10)):  # Show first 10 modes
    print(f"  {j:3d}", end="")
print()

max_diag_error = 0
max_offdiag = 0

for i in range(min(n_modes, 10)):
    print(f"{i:3d}: ", end="")
    for j in range(min(n_modes, 10)):
        inner = np.sum(
            masses_repeated * all_modes_normalized[i].conj() * all_modes_normalized[j]
        )
        print(f"{inner.real:5.2f}", end=" ")

        if i == j:
            max_diag_error = max(max_diag_error, abs(inner.real - 1.0))
        else:
            max_offdiag = max(max_offdiag, abs(inner.real))
    print()

print(f"\nMax diagonal error: {max_diag_error:.6e}")
print(f"Max off-diagonal: {max_offdiag:.6e}")

if max_diag_error > 1e-6:
    print("\n⚠️  PROBLEM: Diagonal elements should be 1.0 but have significant error!")
    print("   This means normalized modes don't have unit norm.")
    print("   Something is wrong with the normalization procedure.")
elif max_offdiag > 1e-6:
    print(
        "\n⚠️  PROBLEM: Off-diagonal elements should be 0.0 but have significant values!"
    )
    print("   This means modes are not orthogonal to each other.")
else:
    print("\n✓ Modes appear orthonormal with small numerical errors.")

print("=" * 80)
