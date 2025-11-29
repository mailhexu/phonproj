"""
Test if phase-scan at phase=0 matches standard decomposition.

This is the critical test: if phase-scan doesn't match standard decomposition
at phase=0, then the implementation is fundamentally wrong.

Run:
    uv run python agent_files/debug/phase_scan/test_phase_zero_match.py
"""

import numpy as np
from phonproj.modes import PhononModes
from phonproj.core.structure_analysis import (
    decompose_displacement_to_modes,
    project_displacement_with_phase_scan,
)

# Load phonon data
yaml_file = "data/BaTiO3_phonopy_params.yaml"

# Use 1x1x1 supercell which only needs Gamma point
supercell_matrix = np.eye(3)  # 1x1x1 - only needs Gamma

qpoints = np.array([[0.0, 0.0, 0.0]])  # Gamma point only
phonon_modes = PhononModes.from_phonopy_yaml(yaml_file, qpoints=qpoints)

# Generate a test displacement: a single mode pattern
test_q_index = 0  # Gamma point
test_mode_index = 3  # Some non-acoustic mode

# Generate the test displacement using the standard method
all_displacements = phonon_modes.generate_all_commensurate_displacements(
    supercell_matrix
)
test_displacement = all_displacements[test_q_index][test_mode_index]

print(f"Testing phase-scan at phase=0 vs standard decomposition")
print(f"Test mode: q_index={test_q_index}, mode_index={test_mode_index}")
print(f"Test displacement shape: {test_displacement.shape}")
print()

# Standard decomposition (take real part since that's what we typically use)
target_displacement_real = test_displacement.real
projection_table, summary = decompose_displacement_to_modes(
    target_displacement_real, phonon_modes, supercell_matrix, normalize=True
)

# Find the projection for our test mode in standard decomposition
standard_coeff = None
for row in projection_table:
    if row["q_index"] == test_q_index and row["mode_index"] == test_mode_index:
        standard_coeff = row["projection_coefficient"]
        break

print(f"Standard decomposition result for test mode: {standard_coeff:.6f}")
print()

# Phase-scan at phase=0
max_coeff, optimal_phase = project_displacement_with_phase_scan(
    target_displacement=target_displacement_real,
    phonon_modes=phonon_modes,
    q_index=test_q_index,
    mode_index=test_mode_index,
    supercell_matrix=supercell_matrix,
    n_phases=11,  # Include 0, π/10, 2π/10, ..., π
)

print(f"Phase-scan result:")
print(f"  Max coefficient: {max_coeff:.6f}")
print(f"  Optimal phase: {optimal_phase:.6f} rad = {np.degrees(optimal_phase):.2f}°")
print()

# The coefficient at phase=0 should be the standard decomposition result
# Let's explicitly calculate it at phase=0
phases = np.linspace(0, np.pi, 11, endpoint=True)
print(f"Phase values tested: {[f'{p:.4f}' for p in phases]}")
print()

# Check if they match
if abs(max_coeff - abs(standard_coeff)) < 1e-6:
    print("✓ Phase-scan matches standard decomposition!")
else:
    print(f"✗ MISMATCH:")
    print(f"  Standard: {standard_coeff:.6f}")
    print(f"  Phase-scan max: {max_coeff:.6f}")
    print(f"  Difference: {abs(max_coeff - abs(standard_coeff)):.6e}")
    print()
    print("This indicates the phase-scan is using a different displacement")
    print("pattern than the standard decomposition.")

print()
print("=" * 60)
print("Testing all modes at Gamma point")
print("=" * 60)

# Test all modes at Gamma point
for mode_idx in range(phonon_modes.frequencies.shape[1]):
    # Get standard decomposition coefficient
    std_coeff = None
    for row in projection_table:
        if row["q_index"] == test_q_index and row["mode_index"] == mode_idx:
            std_coeff = row["projection_coefficient"]
            break

    # Get phase-scan result
    max_coeff_ps, _ = project_displacement_with_phase_scan(
        target_displacement=target_displacement_real,
        phonon_modes=phonon_modes,
        q_index=test_q_index,
        mode_index=mode_idx,
        supercell_matrix=supercell_matrix,
        n_phases=11,
    )

    match = "✓" if abs(max_coeff_ps - abs(std_coeff)) < 1e-6 else "✗"
    print(
        f"{match} Mode {mode_idx:2d}: std={std_coeff:8.6f}, phase-scan={max_coeff_ps:8.6f}, diff={abs(max_coeff_ps - abs(std_coeff)):8.2e}"
    )
