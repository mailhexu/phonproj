"""
Test that phase-scan now matches standard decomposition after normalization fix.

This tests the CLI-level behavior after fixing the normalization issue.

Run:
    uv run python agent_files/debug/phase_scan/test_cli_fix.py
"""

import numpy as np
from phonproj.modes import PhononModes
from phonproj.cli import analyze_displacement, analyze_phase_scan

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

# Scale by arbitrary amount to simulate unnormalized displacement
test_displacement = test_displacement * 3.7

# Flatten for CLI functions
displacement_vector = test_displacement.ravel()

# Prepare phonopy_data dict (as CLI does)
from phonopy import load

phonopy_obj = load(yaml_file)
phonopy_data = {
    "phonopy": phonopy_obj,
    "primitive_cell": phonopy_obj.primitive,
    "phonopy_yaml": yaml_file,
}

print("=" * 70)
print("Testing CLI-level phase-scan vs standard decomposition")
print("=" * 70)
print(
    f"Test: Single mode displacement (q_index={test_q_index}, mode={test_mode_index})"
)
print(f"Scaled by factor 3.7 to simulate unnormalized input")
print()

# Run standard decomposition (suppressing output)
print("Running standard decomposition...")
try:
    from phonproj.core.structure_analysis import decompose_displacement_to_modes

    projection_table, summary = decompose_displacement_to_modes(
        displacement=test_displacement,
        phonon_modes=phonon_modes,
        supercell_matrix=supercell_matrix,
        normalize=True,  # Let it normalize
        sort_by_contribution=False,
    )

    # Find coefficient for our test mode
    std_coeff = None
    for row in projection_table:
        if row["q_index"] == test_q_index and row["mode_index"] == test_mode_index:
            std_coeff = row["projection_coefficient"]
            break

    print(f"Standard decomposition coefficient: {std_coeff:.6f}")
    print(f"Sum of squared projections: {summary['sum_squared_projections']:.6f}")
except Exception as e:
    print(f"Error: {e}")
    std_coeff = None

print()

# Run phase-scan (suppressing output)
print("Running phase-scan...")
try:
    from phonproj.core.structure_analysis import project_displacement_with_phase_scan
    from phonproj.modes import create_supercell

    # Manual normalization (as the fixed analyze_phase_scan does)
    supercell = create_supercell(phonon_modes.primitive_cell, supercell_matrix)
    masses = np.repeat(supercell.get_masses(), 3)
    disp_flat = test_displacement.ravel()
    mass_weighted_norm = np.sqrt(np.sum(masses * disp_flat * disp_flat))
    normalized_displacement = test_displacement / mass_weighted_norm

    print(f"Original mass-weighted norm: {mass_weighted_norm:.6f}")
    print(
        f"After normalization: {np.sqrt(np.sum(masses * normalized_displacement.ravel() ** 2)):.6f}"
    )

    max_coeff, optimal_phase = project_displacement_with_phase_scan(
        target_displacement=normalized_displacement,
        phonon_modes=phonon_modes,
        q_index=test_q_index,
        mode_index=test_mode_index,
        supercell_matrix=supercell_matrix,
        n_phases=37,  # Use more phases for better resolution
    )

    print(f"Phase-scan coefficient: {max_coeff:.6f}")
    print(f"Optimal phase: {optimal_phase:.4f} rad ({np.degrees(optimal_phase):.2f}°)")
except Exception as e:
    print(f"Error: {e}")
    max_coeff = None

print()

# Compare
if std_coeff is not None and max_coeff is not None:
    print("=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"Standard decomposition: {std_coeff:.6f}")
    print(f"Phase-scan:             {max_coeff:.6f}")
    print(f"Difference:             {abs(max_coeff - abs(std_coeff)):.2e}")

    if abs(max_coeff - abs(std_coeff)) < 1e-4:
        print()
        print("✓ SUCCESS: Phase-scan matches standard decomposition!")
    else:
        print()
        print("✗ FAIL: Phase-scan does NOT match standard decomposition")
        print()
        print("This suggests the normalization fix didn't work as expected.")
