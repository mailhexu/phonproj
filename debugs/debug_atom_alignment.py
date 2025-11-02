#!/usr/bin/env python3
"""
Debug script to check atom positions and ordering between structures.
"""

import numpy as np
from ase.io import read
from ase.build import make_supercell

# Load structures
displaced_file = "data/yajundata/CONTCAR-a1a2-GS"
primitive_file = "data/yajundata/0.02-P4mmm-PTO/POSCAR"

print("=" * 80)
print("STRUCTURE ALIGNMENT ANALYSIS")
print("=" * 80)

displaced = read(displaced_file, format="vasp")
primitive = read(primitive_file, format="vasp")

if isinstance(displaced, list):
    displaced = displaced[0]
if isinstance(primitive, list):
    primitive = primitive[0]

# Generate reference supercell
supercell_matrix = np.array([[16, 0, 0], [0, 1, 0], [0, 0, 1]])
reference = make_supercell(primitive, supercell_matrix)

print(f"\nStructure info:")
print(f"  Reference: {len(reference)} atoms")
print(f"  Displaced: {len(displaced)} atoms")
print(
    f"  Chemical symbols match: {reference.get_chemical_symbols() == displaced.get_chemical_symbols()}"
)

# Check cells
ref_cell = reference.get_cell().array
disp_cell = displaced.get_cell().array

print(f"\nCell comparison:")
print(f"  Reference cell:")
for i, vec in enumerate(ref_cell):
    print(f"    {['a', 'b', 'c'][i]}: {vec}")
print(f"  Displaced cell:")
for i, vec in enumerate(disp_cell):
    print(f"    {['a', 'b', 'c'][i]}: {vec}")
print(f"  Max cell difference: {np.max(np.abs(ref_cell - disp_cell)):.6f} Å")

# Get positions
ref_pos = reference.get_positions()
disp_pos = displaced.get_positions()

# Calculate raw displacement WITHOUT PBC wrapping
raw_displacement = disp_pos - ref_pos
raw_norm = np.linalg.norm(raw_displacement)

print(f"\nRAW displacement (no PBC correction):")
print(f"  Total norm: {raw_norm:.3f} Å")
print(f"  Mean per-atom displacement: {raw_norm / len(reference):.3f} Å")
print(f"  Max displacement: {np.max(np.linalg.norm(raw_displacement, axis=1)):.3f} Å")

# Apply PBC wrapping
cell_inv = np.linalg.pinv(ref_cell)
displacement_frac = np.dot(raw_displacement, cell_inv.T)
displacement_frac_wrapped = displacement_frac - np.round(displacement_frac)
displacement_pbc = np.dot(displacement_frac_wrapped, ref_cell)

pbc_norm = np.linalg.norm(displacement_pbc)

print(f"\nPBC-wrapped displacement:")
print(f"  Total norm: {pbc_norm:.3f} Å")
print(f"  Mean per-atom displacement: {pbc_norm / len(reference):.3f} Å")
print(f"  Max displacement: {np.max(np.linalg.norm(displacement_pbc, axis=1)):.3f} Å")

# Check if there are large jumps in fractional coordinates
print(f"\nFractional coordinate wrapping:")
print(
    f"  Atoms wrapped by >=0.5: {np.sum(np.abs(displacement_frac - displacement_frac_wrapped) > 0.1)}"
)
print(f"  Max fractional shift: {np.max(np.abs(np.round(displacement_frac))):.1f}")

# Check atom ordering by comparing element types
ref_symbols = np.array(reference.get_chemical_symbols())
disp_symbols = np.array(displaced.get_chemical_symbols())

if not np.all(ref_symbols == disp_symbols):
    print(f"\n⚠️  ATOM ORDERING MISMATCH DETECTED!")
    print(f"  Atoms are in different order between structures")

    # Count mismatches by element
    for element in set(ref_symbols):
        ref_indices = np.where(ref_symbols == element)[0]
        disp_indices = np.where(disp_symbols == element)[0]
        if not np.array_equal(ref_indices, disp_indices):
            print(f"  {element}: positions differ")
else:
    print(f"\n✓ Atom ordering matches (same element sequence)")

# Analyze displacement distribution
print(f"\nDisplacement statistics (PBC-wrapped, Cartesian):")
per_atom_disp = np.linalg.norm(displacement_pbc, axis=1)
print(f"  Min: {np.min(per_atom_disp):.3f} Å")
print(f"  Mean: {np.mean(per_atom_disp):.3f} Å")
print(f"  Median: {np.median(per_atom_disp):.3f} Å")
print(f"  Max: {np.max(per_atom_disp):.3f} Å")
print(f"  Std: {np.std(per_atom_disp):.3f} Å")

# Check for outliers (atoms displaced > 1 Å)
large_displacements = per_atom_disp > 1.0
if np.any(large_displacements):
    print(f"\n⚠️  {np.sum(large_displacements)} atoms displaced > 1.0 Å")
    large_indices = np.where(large_displacements)[0]
    for idx in large_indices[:10]:  # Show first 10
        print(f"  Atom {idx} ({ref_symbols[idx]}): {per_atom_disp[idx]:.3f} Å")

# Try to find optimal atom mapping
print(f"\n" + "=" * 80)
print("ATOM MAPPING ANALYSIS")
print("=" * 80)

# For each atom in displaced structure, find nearest atom in reference
# This will tell us if atoms need reordering
from scipy.spatial.distance import cdist


# Calculate pairwise distances (accounting for PBC)
def calc_pbc_distances(pos1, pos2, cell):
    """Calculate minimum image distances between two sets of positions."""
    cell_inv = np.linalg.pinv(cell)
    n1 = len(pos1)
    n2 = len(pos2)
    min_dists = np.zeros((n1, n2))

    for i in range(n1):
        for j in range(n2):
            diff = pos2[j] - pos1[i]
            diff_frac = np.dot(diff, cell_inv.T)
            diff_frac = diff_frac - np.round(diff_frac)
            diff_cart = np.dot(diff_frac, cell)
            min_dists[i, j] = np.linalg.norm(diff_cart)

    return min_dists


print(f"\nCalculating optimal atom mapping...")
distances = calc_pbc_distances(ref_pos, disp_pos, ref_cell)

# For each reference atom, find closest displaced atom
closest_matches = np.argmin(distances, axis=1)
closest_distances = np.min(distances, axis=1)

# Check if mapping is identity (i.e., atoms already ordered correctly)
is_identity = np.all(closest_matches == np.arange(len(reference)))

print(f"  Identity mapping: {is_identity}")
print(f"  Mean nearest-neighbor distance: {np.mean(closest_distances):.3f} Å")
print(f"  Max nearest-neighbor distance: {np.max(closest_distances):.3f} Å")

if not is_identity:
    print(f"\n⚠️  ATOM REORDERING REQUIRED!")
    print(
        f"  Number of atoms with wrong pairing: {np.sum(closest_matches != np.arange(len(reference)))}"
    )

    # Show first few mismatches
    mismatches = np.where(closest_matches != np.arange(len(reference)))[0]
    print(f"  First 10 mismatches:")
    for i in mismatches[:10]:
        print(
            f"    Ref atom {i} → Disp atom {closest_matches[i]} (dist: {closest_distances[i]:.3f} Å)"
        )

# Calculate displacement with optimal mapping
reordered_disp_pos = disp_pos[closest_matches]
optimal_displacement = reordered_disp_pos - ref_pos

# Apply PBC wrapping to optimal displacement
displacement_frac_opt = np.dot(optimal_displacement, cell_inv.T)
displacement_frac_opt_wrapped = displacement_frac_opt - np.round(displacement_frac_opt)
displacement_opt_pbc = np.dot(displacement_frac_opt_wrapped, ref_cell)

optimal_norm = np.linalg.norm(displacement_opt_pbc)

print(f"\n" + "=" * 80)
print(f"OPTIMALLY REORDERED displacement:")
print(f"=" * 80)
print(f"  Total norm: {optimal_norm:.3f} Å")
print(f"  Mean per-atom: {optimal_norm / len(reference):.3f} Å")
print(
    f"  Max displacement: {np.max(np.linalg.norm(displacement_opt_pbc, axis=1)):.3f} Å"
)

per_atom_opt = np.linalg.norm(displacement_opt_pbc, axis=1)
print(f"  Min: {np.min(per_atom_opt):.3f} Å")
print(f"  Mean: {np.mean(per_atom_opt):.3f} Å")
print(f"  Median: {np.median(per_atom_opt):.3f} Å")
print(f"  Max: {np.max(per_atom_opt):.3f} Å")

print(f"\n" + "=" * 80)
print(f"COMPARISON:")
print(f"=" * 80)
print(f"  Current method: {pbc_norm:.3f} Å")
print(f"  Optimal reordering: {optimal_norm:.3f} Å")
print(
    f"  Improvement: {(pbc_norm - optimal_norm):.3f} Å ({100 * (pbc_norm - optimal_norm) / pbc_norm:.1f}%)"
)

print(f"\n" + "=" * 80)
