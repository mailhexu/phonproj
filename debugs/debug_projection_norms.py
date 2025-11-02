#!/usr/bin/env python3
"""
Debug script to check normalization of displacement vectors and eigenvectors.
"""

import numpy as np
from phonproj.modes import PhononModes


def generate_16x1x1_qpoints():
    """Generate q-points for 16x1x1 supercell."""
    qpoints = []
    for i in range(16):
        qx = i / 16.0
        qpoints.append([qx, 0, 0])
    return np.array(qpoints)


# Load the phonon data
phonopy_dir = "data/yajundata/0.02-P4mmm-PTO/"
qpoints_16x1x1 = generate_16x1x1_qpoints()

phonon_modes = PhononModes.from_phonopy_directory(phonopy_dir, qpoints=qpoints_16x1x1)

print("=" * 80)
print("CHECKING EIGENVECTOR NORMALIZATION")
print("=" * 80)

# Generate eigenvectors for 16x1x1 supercell at Gamma point
supercell_matrix = np.array([[16, 0, 0], [0, 1, 0], [0, 0, 1]])

# Get commensurate displacements
all_commensurate_displacements = phonon_modes.generate_all_commensurate_displacements(
    supercell_matrix, amplitude=1.0
)

# Check Gamma point (should be index 0)
gamma_idx = 0
gamma_modes = all_commensurate_displacements[gamma_idx]

print(f"\nGamma point modes shape: {gamma_modes.shape}")
print(f"Number of modes at Gamma: {gamma_modes.shape[0]}")

# Get supercell masses
det = int(np.round(np.linalg.det(supercell_matrix)))
supercell_masses = np.tile(phonon_modes.atomic_masses, det)

print(f"\nSupercell has {det} unit cells, {len(supercell_masses)} atoms")

# Check normalization of first few modes
print(f"\n{'Mode':<10} {'Mass-weighted norm':<25} {'Cartesian norm':<25}")
print("-" * 60)

for mode_idx in range(min(6, len(gamma_modes))):
    mode_disp = gamma_modes[mode_idx]  # shape: (n_atoms, 3)

    # Calculate mass-weighted norm: sqrt(sum(m * |d|^2))
    mass_weighted_norm = phonon_modes.mass_weighted_norm(mode_disp, supercell_masses)

    # Calculate cartesian norm: sqrt(sum(|d|^2))
    cartesian_norm = np.sqrt(np.sum(mode_disp**2))

    print(f"{mode_idx:<10} {mass_weighted_norm:<25.10f} {cartesian_norm:<25.10f}")

print("\n" + "=" * 80)
print("CHECKING DISPLACEMENT VECTOR NORMALIZATION")
print("=" * 80)

# Now load the actual displacement from Yajun's data
from ase.io import read
from ase.build import make_supercell

# Load structures
displaced_file = "data/yajundata/CONTCAR-a1a2-GS"
primitive_file = "data/yajundata/0.02-P4mmm-PTO/POSCAR"

print(f"\nLoading structures...")
print(f"  Displaced: {displaced_file}")
print(f"  Primitive: {primitive_file}")

displaced = read(displaced_file, format="vasp")
primitive = read(primitive_file, format="vasp")

# Handle if read returns list
if isinstance(displaced, list):
    displaced = displaced[0]
if isinstance(primitive, list):
    primitive = primitive[0]

# Generate 16x1x1 reference supercell
reference = make_supercell(primitive, supercell_matrix)

print(f"\nReference structure: {len(reference)} atoms")
print(f"Displaced structure: {len(displaced)} atoms")

# Calculate displacement
ref_positions = reference.get_positions()
disp_positions = displaced.get_positions()
displacement_cart = disp_positions - ref_positions

# Handle PBC
cell = reference.get_cell()
if np.any(cell):
    cell_inv = np.linalg.pinv(cell)
    displacement_frac = np.dot(displacement_cart, cell_inv.T)
    displacement_frac = displacement_frac - np.round(displacement_frac)
    displacement_cart = np.dot(displacement_frac, cell)

# Apply mass weighting
masses = reference.get_masses()
mass_weights = np.sqrt(masses)
displacement_mass_weighted = displacement_cart / mass_weights[:, np.newaxis]

# Calculate norms
mw_norm = phonon_modes.mass_weighted_norm(displacement_mass_weighted, masses)
cart_norm = np.sqrt(np.sum(displacement_cart**2))
mw_cart_norm = np.sqrt(np.sum(displacement_mass_weighted**2))

print(f"\nDisplacement vector norms:")
print(f"  - Cartesian norm: {cart_norm:.10f}")
print(f"  - Mass-weighted cartesian norm: {mw_cart_norm:.10f}")
print(f"  - Mass-weighted phonon norm: {mw_norm:.10f}")

print("\n" + "=" * 80)
print("CHECKING SELF-PROJECTION")
print("=" * 80)

# Test self-projection of eigenvector
mode_0 = gamma_modes[0]
self_proj = phonon_modes.mass_weighted_projection(mode_0, mode_0, supercell_masses)
print(f"\nMode 0 self-projection: {abs(self_proj):.10f}")

# Test projection of displacement onto mode 5 (which shows 16.6 in the output)
mode_5 = gamma_modes[5]
displacement_reshaped = displacement_mass_weighted.reshape(-1, 3)

# Check if shapes match
print(f"\nDisplacement shape: {displacement_reshaped.shape}")
print(f"Mode 5 shape: {mode_5.shape}")

if displacement_reshaped.shape == mode_5.shape:
    projection_5 = phonon_modes.mass_weighted_projection(
        displacement_reshaped, mode_5, supercell_masses
    )
    print(f"Projection of displacement onto mode 5: {abs(projection_5):.10f}")

    # Check norm of displacement
    disp_norm = phonon_modes.mass_weighted_norm(displacement_reshaped, supercell_masses)
    print(f"Displacement mass-weighted norm: {disp_norm:.10f}")

    # The projection should be <= disp_norm (by Cauchy-Schwarz)
    print(f"\nCauchy-Schwarz check: |projection| <= ||displacement|| ?")
    print(
        f"  {abs(projection_5):.6f} <= {disp_norm:.6f} ? {abs(projection_5) <= disp_norm}"
    )

    print("\n" + "=" * 80)
    print("UNDERSTANDING PROJECTIONS")
    print("=" * 80)

    # Check projections for several modes
    print(f"\nDisplacement mass-weighted norm: ||u|| = {disp_norm:.6f}")
    print(f"Displacement norm squared: ||u||² = {disp_norm**2:.6f}")

    print(f"\n{'Mode':<8} {'<e|u>':<15} {'|<e|u>|²':<15} {'Normalized c':<15}")
    print("-" * 60)

    total_proj_sq = 0.0
    for mode_idx in [0, 1, 2, 3, 4, 5]:
        mode_disp = gamma_modes[mode_idx]
        projection = phonon_modes.mass_weighted_projection(
            displacement_reshaped, mode_disp, supercell_masses
        )
        proj_abs = abs(projection)
        proj_sq = proj_abs**2
        normalized_coeff = proj_abs / disp_norm

        total_proj_sq += proj_sq

        print(
            f"{mode_idx:<8} {proj_abs:<15.6f} {proj_sq:<15.6f} {normalized_coeff:<15.6f}"
        )

    print("\n" + "=" * 80)
    print("PARSEVAL'S THEOREM CHECK (for first 6 modes only)")
    print("=" * 80)
    print(f"∑|<e|u>|² (first 6 modes): {total_proj_sq:.6f}")
    print(f"||u||²: {disp_norm**2:.6f}")
    print(f"Ratio: {total_proj_sq / (disp_norm**2):.6f}")
    print("\nNote: This ratio should approach 1.0 when summing over ALL 480 modes.")

else:
    print(f"❌ Shape mismatch!")

print("\n" + "=" * 80)
