#!/usr/bin/env python3

import numpy as np
from pathlib import Path
from phonproj.modes import PhononModes

# Load BaTiO3 data
qpoints_2x2x2 = []
for i in range(2):
    for j in range(2):
        for k in range(2):
            qpoints_2x2x2.append([i / 2.0, j / 2.0, k / 2.0])
qpoints_2x2x2 = np.array(qpoints_2x2x2)

BATIO3_YAML_PATH = Path("data/BaTiO3_phonopy_params.yaml")
modes = PhononModes.from_phonopy_yaml(str(BATIO3_YAML_PATH), qpoints_2x2x2)

# Focus on problematic q-points: 1 and 2
problematic_qpoints = [1, 2]  # [0, 0, 0.5] and [0, 0.5, 0]
supercell_matrix = np.eye(3, dtype=int) * 2

print("Detailed analysis of problematic q-points")
print("=" * 60)

for q_idx in problematic_qpoints:
    qpoint = qpoints_2x2x2[q_idx]
    print(f"\nQ-point {q_idx}: {qpoint}")

    # Get the first mode (which fails)
    failing_mode_idx = 0
    eigenvector = modes.eigenvectors[q_idx][:, failing_mode_idx]

    print(f"Mode {failing_mode_idx} eigenvector shape: {eigenvector.shape}")
    print(f"Mode {failing_mode_idx} eigenvector norm: {np.linalg.norm(eigenvector)}")

    # Manual calculation of supercell displacements
    print(f"\nManual phase calculation for first few atoms:")

    # Get primitive positions
    prim_positions = modes.primitive_cell.get_scaled_positions()
    print(f"Primitive positions shape: {prim_positions.shape}")

    # Calculate displacements manually for first few supercell atoms
    n_atoms_primitive = modes._n_atoms
    supercell_size = 8  # 2x2x2

    for supercell_atom_idx in range(min(8, supercell_size * n_atoms_primitive)):
        print(f"\n  Supercell atom {supercell_atom_idx}:")

        # Map supercell atom to primitive atom and lattice vector
        prim_atom_index = supercell_atom_idx % n_atoms_primitive
        supercell_replica = supercell_atom_idx // n_atoms_primitive

        # Convert to 3D lattice coordinates (this is what we fixed)
        nx = ny = nz = 2  # 2x2x2 supercell
        ix = supercell_replica % nx
        iy = (supercell_replica // nx) % ny
        iz = supercell_replica // (nx * ny)
        lattice_vector = np.array([ix, iy, iz], dtype=float)

        print(
            f"    Maps to prim atom {prim_atom_index}, lattice vector {lattice_vector}"
        )

        # Get primitive position
        prim_scaled_pos = prim_positions[prim_atom_index]
        print(f"    Primitive scaled position: {prim_scaled_pos}")

        # Calculate phase factor
        phase_arg = np.dot(qpoint, prim_scaled_pos + lattice_vector)
        phase_factor = np.exp(2j * np.pi * phase_arg)

        print(f"    Phase argument (q·(r+R)): {phase_arg}")
        print(f"    Phase factor exp(2πi*{phase_arg:.3f}): {phase_factor}")
        print(f"    Phase factor magnitude: {abs(phase_factor)}")
        print(f"    Phase factor angle: {np.angle(phase_factor):.3f} rad")

        # Get eigenvector components
        start_idx = prim_atom_index * 3
        end_idx = start_idx + 3
        atom_eigenvector = eigenvector[start_idx:end_idx]

        print(f"    Eigenvector components: {atom_eigenvector}")
        print(f"    Eigenvector magnitude: {np.linalg.norm(atom_eigenvector)}")

        # Apply phase factor
        displacement = atom_eigenvector * phase_factor

        print(f"    Displacement before gauge: {displacement}")
        print(f"    Displacement magnitude: {np.linalg.norm(displacement)}")

        # Apply gauge (assuming "R" gauge)
        displacement_real = displacement.real
        print(f"    Displacement after gauge (real): {displacement_real}")
        print(f"    Final displacement magnitude: {np.linalg.norm(displacement_real)}")

print(f"\n" + "=" * 60)
print("Summary: Looking for cancellation patterns...")

# Check if there's a pattern in the phases that would cause cancellation
for q_idx in problematic_qpoints:
    qpoint = qpoints_2x2x2[q_idx]
    print(f"\nQ-point {q_idx}: {qpoint}")
    print("Phase factors for each unit cell:")

    for cell_idx in range(8):  # 2x2x2 = 8 unit cells
        nx = ny = nz = 2
        ix = cell_idx % nx
        iy = (cell_idx // nx) % ny
        iz = cell_idx // (nx * ny)
        lattice_vector = np.array([ix, iy, iz], dtype=float)

        phase_arg = np.dot(qpoint, lattice_vector)  # Just lattice vector part
        phase_factor = np.exp(2j * np.pi * phase_arg)

        print(
            f"  Cell {cell_idx} {lattice_vector}: exp(2πi*{phase_arg:.3f}) = {phase_factor:.3f}"
        )

print(f"\n" + "=" * 60)
print("Working q-points comparison:")

working_qpoints = [0, 3]  # Examples that work
for q_idx in working_qpoints:
    qpoint = qpoints_2x2x2[q_idx]
    print(f"\nQ-point {q_idx}: {qpoint}")
    print("Phase factors for each unit cell:")

    for cell_idx in range(8):
        nx = ny = nz = 2
        ix = cell_idx % nx
        iy = (cell_idx // nx) % ny
        iz = cell_idx // (nx * ny)
        lattice_vector = np.array([ix, iy, iz], dtype=float)

        phase_arg = np.dot(qpoint, lattice_vector)
        phase_factor = np.exp(2j * np.pi * phase_arg)

        print(
            f"  Cell {cell_idx} {lattice_vector}: exp(2πi*{phase_arg:.3f}) = {phase_factor:.3f}"
        )
