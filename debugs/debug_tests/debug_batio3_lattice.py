#!/usr/bin/env python3

import numpy as np
import os
from pathlib import Path
from phonproj.modes import PhononModes

# Load test data like the actual test
BATIO3_YAML_PATH = Path("/Users/hexu/projects/phonproj/data/BaTiO3_phonopy_params.yaml")

# Set up q-points for 2x2x2
qpoints_2x2x2 = []
for i in range(2):
    for j in range(2):
        for k in range(2):
            qpoints_2x2x2.append([i / 2.0, j / 2.0, k / 2.0])
qpoints_2x2x2 = np.array(qpoints_2x2x2)

print("=== BaTiO3 2x2x2 Lattice Vector Debug ===")
print(f"Q-points: {qpoints_2x2x2}")

# Load modes (suppress warnings)
modes = PhononModes.from_phonopy_yaml(str(BATIO3_YAML_PATH), qpoints_2x2x2)

print(f"Primitive cell atoms: {modes._n_atoms}")
print(f"Atomic masses: {modes.atomic_masses}")

# Check a few displacement generations to see what's happening
supercell_matrix = np.eye(3, dtype=int) * 2
n_supercell_atoms = 8 * modes._n_atoms

print(f"\nSupercell matrix: {supercell_matrix}")
print(f"Supercell atoms: {n_supercell_atoms}")

# Test displacement generation for each q-point
print(f"\nTesting displacement generation for each q-point:")

for q_idx in range(len(qpoints_2x2x2)):
    qpoint = qpoints_2x2x2[q_idx]
    print(f"\nQ-point {q_idx}: {qpoint}")

    try:
        # Generate displacement for first mode of this q-point
        displacement = modes._calculate_supercell_displacements(
            q_idx, 0, supercell_matrix, n_supercell_atoms, phase=0.0
        )

        # Check norm
        supercell_masses = np.tile(modes.atomic_masses, 8)
        norm = modes.mass_weighted_norm(displacement, supercell_masses)

        print(f"  Mode 0 norm: {norm}")

        # Check if all components are essentially zero
        max_component = np.max(np.abs(displacement))
        print(f"  Max displacement component: {max_component}")

        if norm < 1e-10:
            print(f"  *** NEAR-ZERO MODE DETECTED ***")

        # Quick check of first few atoms
        print(f"  First 3 atoms: {displacement[:3]}")

    except Exception as e:
        print(f"  Error generating displacement: {e}")

    if q_idx >= 3:  # Only check first few to save space
        print("  ... (checking more would be too verbose)")
        break

print(f"\nDetailed analysis for q-point 1 (may have issues):")
qpoint = qpoints_2x2x2[1]  # [0.5, 0.0, 0.0]
print(f"Q-point: {qpoint}")

# Manual lattice vector calculation check
print(f"Manual lattice vector mapping for 2x2x2:")
nx, ny, nz = 2, 2, 2
for i in range(min(8, n_supercell_atoms)):
    prim_atom_index = i % modes._n_atoms
    supercell_replica = i // modes._n_atoms

    unit_cell_index = supercell_replica
    ix = unit_cell_index % nx
    iy = (unit_cell_index // nx) % ny
    iz = unit_cell_index // (nx * ny)
    lattice_vector = np.array([ix, iy, iz], dtype=float)

    prim_pos = modes._scaled_positions[prim_atom_index]
    total_pos = prim_pos + lattice_vector

    phase_arg = 2 * np.pi * np.dot(qpoint, total_pos)
    phase = np.exp(1j * phase_arg)

    print(
        f"  Atom {i}: replica={supercell_replica}, lattice=({ix},{iy},{iz}), "
        f"total_pos={total_pos}, phase_arg={phase_arg:.3f}, phase={phase:.3f}"
    )
