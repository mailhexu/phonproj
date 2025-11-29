#!/usr/bin/env python3
"""
Test to verify that phase-scan gives at least as large coefficients as standard projection.

This script compares:
1. Standard projection (phase=0)
2. Phase-scan projection (max over multiple phases)

The phase-scan should give coefficients >= standard projection.
"""

import numpy as np
from phonproj.modes import PhononModes
from phonproj.core.io import load_phonopy_data
from phonproj.core.structure_analysis import (
    project_displacement_with_phase_scan,
    decompose_displacement_to_modes,
)
from ase.io import read

# Load phonopy data
phonopy_dir = "/Users/hexu/projects/phonproj/data/yajundata/0.02-P4mmm-PTO"
phonopy_data = load_phonopy_data(phonopy_dir)

# Simple 2x1x1 supercell for speed
supercell_matrix = np.array([[2, 0, 0], [0, 1, 0], [0, 0, 1]])

# Create a simple test displacement (just the first mode at gamma)
phonopy = phonopy_data["phonopy"]
primitive_cell = phonopy_data["primitive_cell"]

# Get modes at gamma point
from phonproj.core.io import _calculate_phonons_at_kpoints

qpoints = np.array([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]])
frequencies, eigenvectors = _calculate_phonons_at_kpoints(phonopy, qpoints)

phonon_modes = PhononModes(
    primitive_cell=primitive_cell,
    qpoints=qpoints,
    frequencies=frequencies,
    eigenvectors=eigenvectors,
    atomic_masses=None,
    gauge="R",
)

# Generate a test displacement from mode 5 at gamma (to avoid acoustic modes)
test_displacement = phonon_modes.generate_mode_displacement(
    q_index=0,
    mode_index=5,
    supercell_matrix=supercell_matrix,
    amplitude=0.1,
    argument=0.0,  # phase = 0 degrees
)

print("=" * 80)
print("PHASE-SCAN FIX VERIFICATION TEST")
print("=" * 80)
print(f"Supercell: 2×1×1")
print(f"Test displacement: Mode 5 at Gamma point with phase=0°")
print(f"Displacement shape: {test_displacement.shape}")
print()

# Method 1: Standard decomposition (phase=0)
print("Method 1: Standard decomposition (fixed phase)")
print("-" * 80)
projection_table, summary = decompose_displacement_to_modes(
    displacement=test_displacement.real.reshape(-1, 3),
    phonon_modes=phonon_modes,
    supercell_matrix=supercell_matrix,
    normalize=False,
    tolerance=1e-6,
    sort_by_contribution=True,
)

# Find the coefficient for mode 5 at gamma
mode5_standard = None
for entry in projection_table:
    if entry["q_index"] == 0 and entry["mode_index"] == 5:
        mode5_standard = entry["projection_coefficient"]
        print(
            f"Mode 5 at Gamma: coeff = {mode5_standard:.6f}, squared = {entry['squared_coefficient']:.6f}"
        )
        break

# Method 2: Phase-scan with 8 phases
print()
print("Method 2: Phase-scan (8 phases from 0 to π)")
print("-" * 80)
max_coeff, optimal_phase = project_displacement_with_phase_scan(
    phonon_modes=phonon_modes,
    target_displacement=test_displacement.real.reshape(-1, 3),
    supercell_matrix=supercell_matrix,
    q_index=0,
    mode_index=5,
    n_phases=8,
)

print(
    f"Mode 5 at Gamma: max_coeff = {max_coeff:.6f}, optimal_phase = {optimal_phase:.3f} rad ({optimal_phase * 180 / np.pi:.1f}°)"
)
print()

# Comparison
print("=" * 80)
print("COMPARISON")
print("=" * 80)
print(f"Standard projection coefficient: {abs(mode5_standard):.6f}")
print(f"Phase-scan max coefficient:      {max_coeff:.6f}")
print()

if max_coeff >= abs(mode5_standard) - 1e-6:
    print("✅ PASS: Phase-scan coefficient >= standard coefficient")
    ratio = max_coeff / abs(mode5_standard) if abs(mode5_standard) > 0 else float("inf")
    print(f"   Ratio: {ratio:.4f}x")
else:
    print("❌ FAIL: Phase-scan coefficient < standard coefficient")
    print(f"   Difference: {max_coeff - abs(mode5_standard):.6f}")
    print("   This indicates a bug in the phase-scan implementation!")
