"""
Test if normalization is causing phase-scan vs standard decomposition mismatch.

Run:
    uv run python agent_files/debug/phase_scan/test_normalization_issue.py
"""

import numpy as np
from phonproj.modes import PhononModes, create_supercell
from phonproj.core.structure_analysis import (
    decompose_displacement_to_modes,
    project_displacement_with_phase_scan,
)

# Load phonon data
yaml_file = "data/BaTiO3_phonopy_params.yaml"
supercell_matrix = np.eye(3)  # 1x1x1
qpoints = np.array([[0.0, 0.0, 0.0]])
phonon_modes = PhononModes.from_phonopy_yaml(yaml_file, qpoints=qpoints)

# Generate a test displacement: a single mode pattern
test_q_index = 0
test_mode_index = 3

all_displacements = phonon_modes.generate_all_commensurate_displacements(
    supercell_matrix
)
test_displacement = all_displacements[test_q_index][test_mode_index].real

# Get the supercell for mass information
supercell = create_supercell(phonon_modes.primitive_cell, supercell_matrix)
masses = np.repeat(supercell.get_masses(), 3)

# Calculate the mass-weighted norm
disp_flat = test_displacement.ravel()
mass_weighted_norm = np.sqrt(np.sum(masses * disp_flat * disp_flat))
print(f"Original displacement mass-weighted norm: {mass_weighted_norm:.6f}")
print()

# Test 1: Normalized displacement
print("=" * 70)
print("TEST 1: Using NORMALIZED displacement")
print("=" * 70)

normalized_displacement = test_displacement / mass_weighted_norm
norm_check = np.sqrt(np.sum(masses * normalized_displacement.ravel() ** 2))
print(f"Normalized displacement mass-weighted norm: {norm_check:.6f}")
print()

# Standard decomposition with normalized displacement
projection_table1, _ = decompose_displacement_to_modes(
    normalized_displacement, phonon_modes, supercell_matrix, normalize=False
)

std_coeff_norm = None
for row in projection_table1:
    if row["q_index"] == test_q_index and row["mode_index"] == test_mode_index:
        std_coeff_norm = row["projection_coefficient"]
        break

print(f"Standard decomposition (normalized): {std_coeff_norm:.6f}")

# Phase-scan with normalized displacement
max_coeff_norm, _ = project_displacement_with_phase_scan(
    target_displacement=normalized_displacement,
    phonon_modes=phonon_modes,
    q_index=test_q_index,
    mode_index=test_mode_index,
    supercell_matrix=supercell_matrix,
    n_phases=11,
)

print(f"Phase-scan (normalized):             {max_coeff_norm:.6f}")
print(f"Match: {abs(max_coeff_norm - abs(std_coeff_norm)) < 1e-6}")
print()

# Test 2: UN-normalized displacement (scaled by 2.5)
print("=" * 70)
print("TEST 2: Using UNNORMALIZED displacement (scaled by 2.5)")
print("=" * 70)

scaled_displacement = test_displacement * 2.5
scaled_norm = np.sqrt(np.sum(masses * scaled_displacement.ravel() ** 2))
print(f"Scaled displacement mass-weighted norm: {scaled_norm:.6f}")
print()

# Standard decomposition with scaled displacement (let it normalize)
projection_table2, _ = decompose_displacement_to_modes(
    scaled_displacement,
    phonon_modes,
    supercell_matrix,
    normalize=True,  # Let it normalize
)

std_coeff_scaled = None
for row in projection_table2:
    if row["q_index"] == test_q_index and row["mode_index"] == test_mode_index:
        std_coeff_scaled = row["projection_coefficient"]
        break

print(f"Standard decomposition (auto-normalized): {std_coeff_scaled:.6f}")

# Phase-scan with scaled displacement (NO normalization)
max_coeff_scaled_no_norm, _ = project_displacement_with_phase_scan(
    target_displacement=scaled_displacement,
    phonon_modes=phonon_modes,
    q_index=test_q_index,
    mode_index=test_mode_index,
    supercell_matrix=supercell_matrix,
    n_phases=11,
)

print(f"Phase-scan (NO normalization):        {max_coeff_scaled_no_norm:.6f}")
print(f"Expected (2.5 * normalized):          {2.5 * max_coeff_norm:.6f}")
print(
    f"Match with std: {abs(max_coeff_scaled_no_norm - abs(std_coeff_scaled * scaled_norm)) < 1e-4}"
)
print()

# Phase-scan with scaled displacement (manually normalized first)
manually_normalized = scaled_displacement / scaled_norm
max_coeff_scaled_manual_norm, _ = project_displacement_with_phase_scan(
    target_displacement=manually_normalized,
    phonon_modes=phonon_modes,
    q_index=test_q_index,
    mode_index=test_mode_index,
    supercell_matrix=supercell_matrix,
    n_phases=11,
)

print(f"Phase-scan (manually normalized):     {max_coeff_scaled_manual_norm:.6f}")
print(
    f"Match with std: {abs(max_coeff_scaled_manual_norm - abs(std_coeff_scaled)) < 1e-6}"
)
print()

print("=" * 70)
print("CONCLUSION")
print("=" * 70)
print("If phase-scan with unnormalized displacement doesn't match standard")
print("decomposition, then analyze_phase_scan needs to normalize the displacement")
print("before calling project_displacement_with_phase_scan!")
