#!/usr/bin/env python3
"""
Debug the mass consistency issue by tracing through a single mode projection.
"""

import numpy as np
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from phonproj.modes import PhononModes
from phonproj.core.structure_analysis import project_displacements_between_supercells
from phonproj.modes import create_supercell


def debug_single_mode_projection():
    """Debug a single mode projection to identify mass inconsistency."""

    print("Loading Yajun PTO phonon data...")

    # Load data (copy from step10)
    data_dir = project_root / "data" / "yajundata" / "0.02-P4mmm-PTO"

    # Simple qpoints for testing
    qpoints_2x2x1 = []
    for i in range(2):
        for j in range(2):
            for k in range(1):
                qpoints_2x2x1.append([i / 2.0, j / 2.0, k / 1.0])
    qpoints = np.array(qpoints_2x2x1)

    print(f"Using {len(qpoints)} commensurate q-points for (2,2,1) supercell")

    # Load phonon modes
    phonon_modes = PhononModes.from_phonopy_directory(str(data_dir), qpoints=qpoints)

    print(
        f"Loaded {phonon_modes.n_qpoints} q-points with {phonon_modes.n_modes} modes each"
    )

    # Create supercells
    supercell_matrix = np.array([[2, 0, 0], [0, 2, 0], [0, 0, 1]])

    # Create target supercell (as done in decompose_displacement_to_modes)
    target_supercell = create_supercell(phonon_modes.primitive_cell, supercell_matrix)
    print(f"Target supercell: {len(target_supercell)} atoms")

    # Create source supercell (as done in decompose_displacement_to_modes loop)
    source_supercell = create_supercell(phonon_modes.primitive_cell, supercell_matrix)
    print(f"Source supercell: {len(source_supercell)} atoms")

    # Check if masses are identical
    target_masses = target_supercell.get_masses()
    source_masses = source_supercell.get_masses()

    print(f"Target masses shape: {target_masses.shape}")
    print(f"Source masses shape: {source_masses.shape}")

    masses_identical = np.allclose(target_masses, source_masses)
    print(f"Masses identical: {masses_identical}")

    if not masses_identical:
        print("PROBLEM: Masses are not identical!")
        diff = np.abs(target_masses - source_masses)
        print(f"Max difference: {np.max(diff)}")
        print(f"First 5 target masses: {target_masses[:5]}")
        print(f"First 5 source masses: {source_masses[:5]}")

        # Check if it's just reordering
        target_sorted = np.sort(target_masses)
        source_sorted = np.sort(source_masses)
        if np.allclose(target_sorted, source_sorted):
            print("It's just a reordering issue")
        else:
            print("Not just reordering - different mass values")

    # Test a single mode displacement
    q_index = 0
    mode_index = 0

    print(f"\nTesting q_index={q_index}, mode_index={mode_index}")

    # Generate mode displacement
    mode_displacement = phonon_modes.generate_mode_displacement(
        q_index=q_index,
        mode_index=mode_index,
        supercell_matrix=supercell_matrix,
        amplitude=1.0,
    )

    print(f"Mode displacement shape: {mode_displacement.shape}")

    # Create a synthetic target displacement (e.g., same as mode displacement)
    target_displacement = mode_displacement.copy()

    # Test projection with both normalization approaches
    print("\n1. Testing with normalize=True:")
    coeff1 = project_displacements_between_supercells(
        source_displacement=mode_displacement,
        target_displacement=target_displacement,
        source_supercell=source_supercell,
        target_supercell=target_supercell,
        normalize=True,
        use_mass_weighting=True,
    )
    print(f"   Coefficient: {coeff1}")

    print(
        "\n2. Testing with normalize=False (manual normalization using source masses):"
    )
    # Normalize mode displacement using source masses
    source_masses_3d = np.repeat(source_masses, 3)
    mode_flat = mode_displacement.ravel()
    mode_norm = np.sqrt(np.sum(source_masses_3d * mode_flat.conj() * mode_flat))
    normalized_mode = mode_displacement / mode_norm

    # Normalize target displacement using target masses
    target_masses_3d = np.repeat(target_masses, 3)
    target_flat = target_displacement.ravel()
    target_norm = np.sqrt(np.sum(target_masses_3d * target_flat.conj() * target_flat))
    normalized_target = target_displacement / target_norm

    coeff2 = project_displacements_between_supercells(
        source_displacement=normalized_mode,
        target_displacement=normalized_target,
        source_supercell=source_supercell,
        target_supercell=target_supercell,
        normalize=False,
        use_mass_weighting=True,
    )
    print(f"   Coefficient: {coeff2}")

    print(
        "\n3. Testing with normalize=False (manual normalization using target masses):"
    )
    # Normalize BOTH using target masses
    mode_norm_target = np.sqrt(np.sum(target_masses_3d * mode_flat.conj() * mode_flat))
    normalized_mode_target = mode_displacement / mode_norm_target

    coeff3 = project_displacements_between_supercells(
        source_displacement=normalized_mode_target,
        target_displacement=normalized_target,
        source_supercell=source_supercell,
        target_supercell=target_supercell,
        normalize=False,
        use_mass_weighting=True,
    )
    print(f"   Coefficient: {coeff3}")

    print(f"\nComparison:")
    print(f"  normalize=True:                {coeff1:.6f}")
    print(f"  manual with source masses:     {coeff2:.6f}")
    print(f"  manual with target masses:     {coeff3:.6f}")

    if not masses_identical:
        print(f"  Difference (1 vs 2): {abs(coeff1 - coeff2):.6f}")
        print(f"  Difference (1 vs 3): {abs(coeff1 - coeff3):.6f}")
        print(f"  Difference (2 vs 3): {abs(coeff2 - coeff3):.6f}")

    # The coefficients should all be ~1.0 for identical displacements
    expected = 1.0
    print(f"\nExpected coefficient: {expected}")
    print(f"Error from normalize=True: {abs(coeff1 - expected):.6f}")
    if not masses_identical:
        print(f"Error from source masses: {abs(coeff2 - expected):.6f}")
        print(f"Error from target masses: {abs(coeff3 - expected):.6f}")


if __name__ == "__main__":
    debug_single_mode_projection()
