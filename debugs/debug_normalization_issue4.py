#!/usr/bin/env python3
"""
Debug part 4: Check if eigenvectors are orthogonal in primitive cell basis.
"""

import numpy as np
from phonproj.modes import PhononModes

# Load BaTiO3 data
gamma_qpoint = np.array([[0.0, 0.0, 0.0]])
modes = PhononModes.from_phonopy_yaml("data/BaTiO3_phonopy_params.yaml", gamma_qpoint)

print("=" * 80)
print("DEBUG PART 4: Eigenvector Orthogonality in Primitive Cell")
print("=" * 80)

# Check eigenvectors in primitive cell
q_index = 0
primitive_masses = modes.primitive_cell.get_masses()
prim_masses_repeated = np.repeat(primitive_masses, 3)

print(f"\nPrimitive cell: {len(modes.primitive_cell)} atoms")
print(f"Number of modes: {modes.n_modes}")

# Get all eigenvectors for this q-point
eigenvectors = modes.eigenvectors[q_index]  # Shape: (n_modes, n_atoms*3)

print(f"Eigenvectors shape: {eigenvectors.shape}")

# Check orthonormality in primitive cell
print(f"\n--- Gram matrix in primitive cell (mass-weighted) ---")
print("     ", end="")
for j in range(min(modes.n_modes, 10)):
    print(f"  {j:3d}", end="")
print()

max_diag_error = 0
max_offdiag = 0

for i in range(min(modes.n_modes, 10)):
    print(f"{i:3d}: ", end="")
    for j in range(min(modes.n_modes, 10)):
        # Mass-weighted inner product in primitive cell
        inner = np.sum(prim_masses_repeated * eigenvectors[i].conj() * eigenvectors[j])
        print(f"{inner.real:5.2f}", end=" ")

        if i == j:
            max_diag_error = max(max_diag_error, abs(inner.real - 1.0))
        else:
            max_offdiag = max(max_offdiag, abs(inner.real))
    print()

print(f"\nMax diagonal error: {max_diag_error:.6e}")
print(f"Max off-diagonal: {max_offdiag:.6e}")

if max_diag_error < 1e-6 and max_offdiag < 1e-6:
    print("\n✓ Eigenvectors ARE orthonormal in primitive cell!")
else:
    print("\n✗ Eigenvectors are NOT orthonormal even in primitive cell!")

print("\n" + "=" * 80)
print("HYPOTHESIS:")
print("If eigenvectors are orthonormal in primitive cell but NOT in supercell,")
print("then the issue is with how we extend eigenvectors to the supercell.")
print()
print("The problem might be:")
print("1. Using supercell masses instead of primitive cell masses")
print("2. The phase factors e^(iq·r) break orthogonality")
print("3. The 1/√(n_cells) normalization is incorrect")
print("=" * 80)
