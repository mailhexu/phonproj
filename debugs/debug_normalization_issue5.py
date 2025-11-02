#!/usr/bin/env python3
"""
Debug part 5: Understand phonopy's eigenvector normalization convention.
"""

import numpy as np
from phonproj.modes import PhononModes

# Load BaTiO3 data
gamma_qpoint = np.array([[0.0, 0.0, 0.0]])
modes = PhononModes.from_phonopy_yaml("data/BaTiO3_phonopy_params.yaml", gamma_qpoint)

print("=" * 80)
print("DEBUG PART 5: Phonopy Eigenvector Normalization Convention")
print("=" * 80)

q_index = 0
eigenvectors = modes.eigenvectors[q_index]  # Shape: (n_modes, 15)
primitive_masses = modes.primitive_cell.get_masses()

print(f"\nPrimitive cell: {len(modes.primitive_cell)} atoms")
print(f"Masses: {primitive_masses}")
print(f"Number of modes: {modes.n_modes}")

# Check different possible normalizations
print(f"\n--- Testing Different Normalization Conventions ---")

for mode_idx in [0, 3, 6, 9, 14]:
    eigvec = eigenvectors[mode_idx]

    # Reshape to (n_atoms, 3)
    eigvec_reshaped = eigvec.reshape(-1, 3)

    # Different possible norms:
    # 1. Simple L2 norm (no mass weighting)
    l2_norm = np.linalg.norm(eigvec)

    # 2. Mass-weighted norm with √m
    prim_masses_repeated = np.repeat(primitive_masses, 3)
    mass_weighted_sqrt = np.sqrt(np.sum(prim_masses_repeated * np.abs(eigvec) ** 2))

    # 3. Mass-weighted norm without √m (just m)
    mass_weighted = np.sqrt(
        np.sum(primitive_masses[:, np.newaxis] * np.abs(eigvec_reshaped) ** 2)
    )

    # 4. No normalization - sum of squared components
    sum_sq = np.sum(np.abs(eigvec) ** 2)

    # 5. Phonopy convention: eigenvectors might be mass-weighted displacements
    #    So to get Cartesian displacement, divide by √m
    cartesian_norm = np.sqrt(
        np.sum((eigvec_reshaped.real / np.sqrt(primitive_masses[:, np.newaxis])) ** 2)
    )

    print(f"\nMode {mode_idx}:")
    print(f"  L2 norm:                  {l2_norm:.6f}")
    print(f"  Mass-weighted (√m):       {mass_weighted_sqrt:.6f}")
    print(f"  Mass-weighted (m):        {mass_weighted:.6f}")
    print(f"  Sum of squares:           {sum_sq:.6f}")
    print(f"  Cartesian norm (÷√m):     {cartesian_norm:.6f}")

# Check if eigenvectors are orthogonal WITHOUT mass weighting
print(f"\n--- Gram Matrix WITHOUT Mass Weighting ---")
print("     ", end="")
for j in range(min(modes.n_modes, 10)):
    print(f"  {j:3d}", end="")
print()

max_diag_val = 0
max_offdiag = 0

for i in range(min(modes.n_modes, 10)):
    print(f"{i:3d}: ", end="")
    for j in range(min(modes.n_modes, 10)):
        # Simple inner product (no mass weighting)
        inner = np.sum(eigenvectors[i].conj() * eigenvectors[j])
        print(f"{inner.real:5.2f}", end=" ")

        if i == j:
            max_diag_val = max(max_diag_val, abs(inner.real))
        else:
            max_offdiag = max(max_offdiag, abs(inner.real))
    print()

print(f"\nMax diagonal value: {max_diag_val:.6f}")
print(f"Max off-diagonal: {max_offdiag:.6f}")

print("\n" + "=" * 80)
print("PHONOPY NORMALIZATION:")
print("Based on the data, phonopy eigenvectors appear to be:")
print("- NOT unit norm")
print("- NOT mass-weighted orthonormal")
print("- Possibly using a different convention entirely")
print()
print("Common conventions:")
print("1. Phonopy: ε = eigenvector is dimensionless amplitude")
print("2. Displacement: u(r) = (1/√m) * ε * e^(iq·r)")
print("3. Normalization: Σ_α |ε_α|² = N (N = number of atoms)")
print("=" * 80)
