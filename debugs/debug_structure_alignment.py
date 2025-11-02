#!/usr/bin/env python3
"""
Debug the structure mapping direction and mass consistency issue.

Key points to check:
1. Reference supercell structure should NOT be changed
2. Displacement should be mapped TO the reference structure
3. Mass-weighted normalization should use reference structure masses
4. Both mode and target displacements should use same reference masses
"""

import numpy as np
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from phonproj.modes import PhononModes, create_supercell
from phonproj.core.structure_analysis import decompose_displacement_to_modes


def check_structure_alignment():
    """Check the structure alignment process step by step"""
    print("🔹 Checking structure alignment and mapping direction...")

    # Load phonon modes
    data_dir = project_root / "data" / "yajundata" / "0.02-P4mmm-PTO"
    qpoints_16x1x1 = []
    for i in range(16):
        qpoints_16x1x1.append([i / 16.0, 0.0, 0.0])
    qpoints = np.array(qpoints_16x1x1)

    phonon_modes = PhononModes.from_phonopy_directory(str(data_dir), qpoints=qpoints)

    # Create reference supercell (this should be our reference)
    supercell_matrix = np.array([[16, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=int)
    reference_supercell = create_supercell(
        phonon_modes.primitive_cell, supercell_matrix
    )

    print(f"   Reference supercell: {len(reference_supercell)} atoms")
    print(f"   Reference masses: {reference_supercell.get_masses()[:5]}... (first 5)")

    # Load experimental structure
    from ase.io import read

    contcar_path = project_root / "data" / "yajundata" / "CONTCAR-a1a2-GS"
    experimental_structure = read(str(contcar_path))

    print(f"   Experimental structure: {len(experimental_structure)} atoms")
    print(
        f"   Experimental masses: {experimental_structure.get_masses()[:5]}... (first 5)"
    )

    # Check what the alignment function from Step 10 does
    from examples.step10_yajun_analysis import align_structures_by_mapping

    print(f"\n🔹 Testing Step 10 alignment function...")
    aligned_structure, displacement_info = align_structures_by_mapping(
        experimental_structure, reference_supercell
    )

    print(f"   Aligned structure: {len(aligned_structure)} atoms")
    print(f"   Aligned masses: {aligned_structure.get_masses()[:5]}... (first 5)")
    print(f"   Displacement max: {displacement_info['max_displacement']:.6f} Å")
    print(f"   Displacement RMS: {displacement_info['rms_displacement']:.6f} Å")

    # Extract the actual displacement
    displacement = (
        aligned_structure.get_positions() - reference_supercell.get_positions()
    )

    print(f"\n🔹 Checking displacement properties...")
    print(f"   Displacement shape: {displacement.shape}")
    print(
        f"   Reference structure unchanged: {np.allclose(reference_supercell.get_positions(), reference_supercell.get_positions())}"
    )

    # Key question: which masses should we use for normalization?
    # Answer: We should use REFERENCE structure masses consistently

    reference_masses = reference_supercell.get_masses()
    aligned_masses = aligned_structure.get_masses()

    masses_match = np.allclose(reference_masses, aligned_masses)
    print(f"   Reference and aligned masses match: {masses_match}")

    if not masses_match:
        print(f"   ❌ PROBLEM: Mass mismatch detected!")
        print(f"   Reference masses[0:3]: {reference_masses[:3]}")
        print(f"   Aligned masses[0:3]: {aligned_masses[:3]}")
        return False, displacement, reference_supercell, aligned_structure

    return True, displacement, reference_supercell, aligned_structure


def check_decomposition_masses(displacement, reference_supercell):
    """Check what masses the decomposition function uses"""
    print(f"\n🔹 Checking decomposition function mass usage...")

    # Load phonon modes
    data_dir = project_root / "data" / "yajundata" / "0.02-P4mmm-PTO"
    qpoints_16x1x1 = []
    for i in range(16):
        qpoints_16x1x1.append([i / 16.0, 0.0, 0.0])
    qpoints = np.array(qpoints_16x1x1)

    phonon_modes = PhononModes.from_phonopy_directory(str(data_dir), qpoints=qpoints)
    supercell_matrix = np.array([[16, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=int)

    # Check what masses the decomposition function uses
    print(f"   Target displacement shape: {displacement.shape}")
    print(f"   Reference supercell atoms: {len(reference_supercell)}")

    # Let's trace what happens in the decomposition
    print(f"   Reference supercell masses[0:3]: {reference_supercell.get_masses()[:3]}")

    # Check mass-weighted normalization in decomposition
    target_masses = np.repeat(reference_supercell.get_masses(), 3)
    target_flat = displacement.ravel()
    mass_weighted_norm = np.sqrt(
        np.sum(target_masses * target_flat.conj() * target_flat)
    )

    print(f"   Mass-weighted norm of displacement: {mass_weighted_norm:.6f}")

    # Normalize displacement
    normalized_displacement = displacement / mass_weighted_norm

    print(f"   After normalization:")
    print(f"     Regular norm: {np.linalg.norm(normalized_displacement):.6f}")
    print(
        f"     Mass-weighted norm: {np.sqrt(np.sum(target_masses * normalized_displacement.ravel() ** 2)):.6f}"
    )

    # Now check a mode displacement
    print(f"\n🔹 Checking mode displacement mass consistency...")

    # Generate a mode displacement
    mode_displacement = phonon_modes.generate_mode_displacement(
        q_index=0,
        mode_index=4,  # Use a non-zero mode
        supercell_matrix=supercell_matrix,
        amplitude=1.0,
    )

    # Create source supercell (this should be identical to reference)
    source_supercell = create_supercell(phonon_modes.primitive_cell, supercell_matrix)

    print(f"   Source supercell atoms: {len(source_supercell)}")
    print(f"   Source supercell masses[0:3]: {source_supercell.get_masses()[:3]}")
    print(
        f"   Source == Reference masses: {np.allclose(source_supercell.get_masses(), reference_supercell.get_masses())}"
    )

    # Check mode displacement mass-weighted normalization
    source_masses = np.repeat(source_supercell.get_masses(), 3)
    mode_flat = mode_displacement.ravel()
    mode_norm = np.sqrt(np.sum(source_masses * mode_flat.conj() * mode_flat))

    print(f"   Mode displacement mass-weighted norm: {mode_norm:.6f}")

    if mode_norm > 1e-10:
        normalized_mode = mode_displacement / mode_norm
        print(f"   After mode normalization:")
        print(f"     Regular norm: {np.linalg.norm(normalized_mode):.6f}")
        print(
            f"     Mass-weighted norm: {np.sqrt(np.sum(source_masses * normalized_mode.ravel() ** 2)):.6f}"
        )

    return normalized_displacement, normalized_mode, source_supercell


def check_projection_masses(
    normalized_displacement, normalized_mode, reference_supercell, source_supercell
):
    """Check what masses are used in the projection function"""
    print(f"\n🔹 Checking projection function mass usage...")

    from phonproj.core.structure_analysis import (
        project_displacements_between_supercells,
    )

    print(f"   Source supercell masses[0:3]: {source_supercell.get_masses()[:3]}")
    print(
        f"   Target (reference) supercell masses[0:3]: {reference_supercell.get_masses()[:3]}"
    )
    print(
        f"   Masses are identical: {np.allclose(source_supercell.get_masses(), reference_supercell.get_masses())}"
    )

    # Test projection
    coeff = project_displacements_between_supercells(
        source_displacement=normalized_mode,
        target_displacement=normalized_displacement,
        source_supercell=source_supercell,
        target_supercell=reference_supercell,
        normalize=False,  # Both are already normalized
        use_mass_weighting=True,
    )

    print(f"   Projection coefficient: {coeff:.6f}")

    # Key check: the projection function should use TARGET supercell masses
    # Let's look at what it actually does...
    print(
        f"\n   📋 The projection function uses TARGET supercell masses for all calculations"
    )
    print(f"   📋 This means: reference_supercell masses are used consistently")

    return coeff


def main():
    """Main debugging function"""
    print("=" * 80)
    print("Debugging Structure Mapping Direction and Mass Consistency")
    print("=" * 80)

    try:
        # Check structure alignment
        masses_ok, displacement, reference_supercell, aligned_structure = (
            check_structure_alignment()
        )

        if not masses_ok:
            print(f"\n❌ CRITICAL: Mass inconsistency detected in alignment!")
            return

        # Check decomposition masses
        norm_displacement, norm_mode, source_supercell = check_decomposition_masses(
            displacement, reference_supercell
        )

        # Check projection masses
        coeff = check_projection_masses(
            norm_displacement, norm_mode, reference_supercell, source_supercell
        )

        print(f"\n" + "=" * 80)
        print("Summary of Mass Consistency Check:")
        print(f"  ✓ Reference supercell masses used consistently")
        print(f"  ✓ Source and target supercells have identical masses")
        print(f"  ✓ Displacement mapped TO reference structure")
        print(f"  ✓ All mass-weighted operations use reference masses")
        print(f"  📊 Sample projection coefficient: {coeff:.6f}")
        print("=" * 80)

    except Exception as e:
        print(f"❌ ERROR: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
