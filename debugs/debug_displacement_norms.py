#!/usr/bin/env python3
"""
Debug displacement normalization issue.
"""

import numpy as np
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from phonproj.modes import PhononModes
from ase.io import read
from phonproj.modes import create_supercell


def main():
    print("🔹 Analyzing displacement normalization issue...")

    # Load phonon modes
    data_dir = project_root / "data" / "yajundata" / "0.02-P4mmm-PTO"
    qpoints_16x1x1 = []
    for i in range(16):
        qpoints_16x1x1.append([i / 16.0, 0.0, 0.0])
    qpoints = np.array(qpoints_16x1x1)

    phonon_modes = PhononModes.from_phonopy_directory(str(data_dir), qpoints=qpoints)

    # Create supercell
    supercell_matrix = np.array([[16, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=int)
    target_supercell = create_supercell(phonon_modes.primitive_cell, supercell_matrix)

    # Load experimental structure
    contcar_path = project_root / "data" / "yajundata" / "CONTCAR-a1a2-GS"
    experimental_structure = read(str(contcar_path))

    # Extract displacement (simplified version of the full pipeline)
    from phonproj.core.structure_analysis import align_structures_by_mapping

    aligned_structure, _ = align_structures_by_mapping(
        experimental_structure, target_supercell
    )

    # Calculate displacement
    displacement = aligned_structure.get_positions() - target_supercell.get_positions()

    print(f"   Original displacement:")
    print(f"     Shape: {displacement.shape}")
    print(f"     Max amplitude: {np.max(np.abs(displacement)):.6f} Å")
    print(f"     RMS: {np.sqrt(np.mean(displacement**2)):.6f} Å")
    print(f"     Regular norm: {np.linalg.norm(displacement):.6f}")

    # Mass-weighted norm
    target_masses = np.repeat(target_supercell.get_masses(), 3)
    displacement_flat = displacement.ravel()
    mass_weighted_norm = np.sqrt(
        np.sum(target_masses * displacement_flat * displacement_flat)
    )

    print(f"     Mass-weighted norm: {mass_weighted_norm:.6f}")

    # After normalization
    normalized_displacement = displacement / mass_weighted_norm
    print(f"   After mass-weighted normalization:")
    print(f"     Regular norm: {np.linalg.norm(normalized_displacement):.6f}")
    print(f"     Max amplitude: {np.max(np.abs(normalized_displacement)):.6f} Å")

    # Check masses
    masses = target_supercell.get_masses()
    print(f"   Atomic masses:")
    print(f"     Min: {np.min(masses):.3f} amu")
    print(f"     Max: {np.max(masses):.3f} amu")
    print(f"     Mean: {np.mean(masses):.3f} amu")

    # The issue might be that after mass-weighted normalization,
    # the displacement becomes very small, so projections are also very small

    # Test with larger amplitude
    print(f"\n🔹 Testing with artificially large displacement...")
    large_displacement = displacement * 100  # Make it 100x larger

    large_mass_weighted_norm = np.sqrt(
        np.sum(target_masses * large_displacement.ravel() * large_displacement.ravel())
    )
    normalized_large = large_displacement / large_mass_weighted_norm

    print(f"   Large displacement (100x):")
    print(f"     Regular norm: {np.linalg.norm(normalized_large):.6f}")
    print(f"     Max amplitude: {np.max(np.abs(normalized_large)):.6f} Å")

    # The key insight: mass-weighted normalization makes the displacement very small
    # because the masses are large (Ba: 137, Pb: 207, Ti: 48, O: 16)

    print(f"\n🔹 Key insight:")
    print(f"   Mass-weighted normalization shrinks large-mass displacements")
    print(f"   This is correct physics but makes projections very small")
    print(f"   The sum should still be 1.0 if completeness holds")


if __name__ == "__main__":
    main()
