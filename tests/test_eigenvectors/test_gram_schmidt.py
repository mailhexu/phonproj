#!/usr/bin/env python3
"""
Test Gram-Schmidt orthogonalization to create proper orthogonal basis.
"""

import numpy as np
from pathlib import Path
from phonproj.modes import PhononModes


def gram_schmidt_orthogonalize(modes_list, mass_weighted_inner_product_func):
    """
    Apply Gram-Schmidt orthogonalization to a list of modes.

    Args:
        modes_list: List of displacement arrays
        mass_weighted_inner_product_func: Function to compute mass-weighted inner product

    Returns:
        List of orthogonalized modes
    """
    orthogonal_modes = []

    for i, mode in enumerate(modes_list):
        # Start with the original mode
        orthogonal_mode = mode.copy()

        # Subtract projections onto all previous orthogonal modes
        for j, prev_orthogonal_mode in enumerate(orthogonal_modes):
            # Compute projection coefficient
            numerator = mass_weighted_inner_product_func(
                orthogonal_mode, prev_orthogonal_mode
            )
            denominator = mass_weighted_inner_product_func(
                prev_orthogonal_mode, prev_orthogonal_mode
            )

            if abs(denominator) > 1e-12:  # Avoid division by zero
                projection_coeff = numerator / denominator
                # Subtract the projection
                orthogonal_mode = (
                    orthogonal_mode - projection_coeff * prev_orthogonal_mode
                )

        # Check if the resulting mode is non-zero
        norm = np.sqrt(
            mass_weighted_inner_product_func(orthogonal_mode, orthogonal_mode).real
        )
        if norm > 1e-12:
            # Normalize the orthogonal mode
            orthogonal_mode = orthogonal_mode / norm
            orthogonal_modes.append(orthogonal_mode)
        else:
            print(f"Mode {i} is linearly dependent, skipping...")

    return orthogonal_modes


def test_gram_schmidt_fix():
    """Test if Gram-Schmidt orthogonalization fixes the completeness issue."""

    # Load BaTiO3 data
    BATIO3_YAML_PATH = Path("data/BaTiO3_phonopy_params.yaml")

    qpoints_2x2x2 = []
    for i in range(2):
        for j in range(2):
            for k in range(2):
                qpoints_2x2x2.append([i / 2.0, j / 2.0, k / 2.0])
    qpoints_2x2x2 = np.array(qpoints_2x2x2)

    modes = PhononModes.from_phonopy_yaml(str(BATIO3_YAML_PATH), qpoints_2x2x2)
    supercell_matrix = np.eye(3, dtype=int) * 2

    print("Testing Gram-Schmidt orthogonalization fix...")

    # Collect all modes
    all_modes = []
    all_displacements = modes.generate_all_commensurate_displacements(
        supercell_matrix, amplitude=1.0
    )

    for q_idx, displacements in all_displacements.items():
        for mode_idx in range(displacements.shape[0]):
            all_modes.append(displacements[mode_idx])

    print(f"Total modes collected: {len(all_modes)}")

    # Define mass-weighted inner product function
    def mass_weighted_inner_product(mode1, mode2):
        return modes.mass_weighted_projection(mode1, mode2)

    # Apply Gram-Schmidt orthogonalization
    print("Applying Gram-Schmidt orthogonalization...")
    orthogonal_modes = gram_schmidt_orthogonalize(
        all_modes, mass_weighted_inner_product
    )

    print(f"Orthogonal modes after Gram-Schmidt: {len(orthogonal_modes)}")
    print(f"Expected for 2x2x2 supercell: {8 * 5 * 3} = 120")

    # Test orthogonality of the result
    print("\nTesting orthogonality of Gram-Schmidt modes...")
    n_modes = len(orthogonal_modes)
    max_off_diagonal = 0.0

    for i in range(min(10, n_modes)):  # Test first 10 modes for speed
        for j in range(i + 1, min(10, n_modes)):
            projection = mass_weighted_inner_product(
                orthogonal_modes[i], orthogonal_modes[j]
            )
            max_off_diagonal = max(max_off_diagonal, abs(projection))

    print(f"Maximum off-diagonal projection: {max_off_diagonal:.2e}")

    # Test completeness with orthogonalized modes
    print("\nTesting completeness with orthogonalized modes...")

    # Create a test displacement (same as used in Step 7)
    test_displacement = np.random.rand(8 * 5, 3)  # Random displacement

    # Project onto orthogonalized modes
    total_projection = 0.0
    for ortho_mode in orthogonal_modes:
        projection = mass_weighted_inner_product(
            test_displacement.ravel(), ortho_mode.ravel()
        )
        total_projection += abs(projection) ** 2

    print(f"Total projection magnitude squared: {total_projection:.6f}")
    print(f"Expected for complete basis: ~1.0")
    print(f"Deviation: {abs(total_projection - 1.0):.6f}")


if __name__ == "__main__":
    test_gram_schmidt_fix()
