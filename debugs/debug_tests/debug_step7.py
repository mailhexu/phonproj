#!/usr/bin/env python3
"""
Debug script for Step 7 implementation.
"""

import numpy as np
from pathlib import Path
from phonproj.modes import PhononModes

# Test data path
BATIO3_YAML_PATH = Path(__file__).parent / "data" / "BaTiO3_phonopy_params.yaml"


def debug_gamma_orthonormality():
    """Debug the Gamma point orthonormality test."""

    # Load BaTiO3 data - only need Gamma point
    gamma_qpoint = np.array([[0.0, 0.0, 0.0]])
    print(f"Loading data from {BATIO3_YAML_PATH}")
    modes = PhononModes.from_phonopy_yaml(str(BATIO3_YAML_PATH), gamma_qpoint)

    print(f"Loaded modes with {modes._n_atoms} atoms")
    print(f"Atomic masses: {modes.atomic_masses}")
    print(f"Q-points: {modes.qpoints}")
    print(f"Frequencies shape: {modes.frequencies.shape}")
    print(f"Eigenvectors shape: {modes.eigenvectors.shape}")

    # Use 1x1x1 supercell (identity matrix)
    supercell_matrix = np.eye(3, dtype=int)
    gamma_index = 0

    print(f"\nGenerating displacements for Gamma point...")

    # Generate displacements for all modes at Gamma point
    all_displacements = modes.generate_all_mode_displacements(
        gamma_index, supercell_matrix, amplitude=1.0
    )

    print(f"Generated displacements shape: {all_displacements.shape}")
    print(f"Number of modes: {all_displacements.shape[0]}")

    # Check a few norms and projections
    print(f"\nChecking first few displacements:")
    for i in range(min(3, all_displacements.shape[0])):
        norm = modes.mass_weighted_norm(all_displacements[i])
        print(f"Mode {i}: norm = {norm}")

        # Check self-projection (should be 1)
        self_projection = modes.mass_weighted_projection(
            all_displacements[i], all_displacements[i]
        )
        print(f"Mode {i}: self-projection = {self_projection}")

    # Check orthogonality between first two modes
    if all_displacements.shape[0] > 1:
        cross_projection = modes.mass_weighted_projection(
            all_displacements[0], all_displacements[1]
        )
        print(f"Cross-projection (0,1): {cross_projection}")

    # Check if we have the right normalization
    print(f"\nChecking normalization...")
    print(f"Expected: Each mode should have mass-weighted norm = 1")
    print(f"Expected: Each self-projection should be 1")
    print(f"Expected: Cross-projections should be 0")


if __name__ == "__main__":
    debug_gamma_orthonormality()
