#!/usr/bin/env python3
"""
Test the hypothesis that we should only use inequivalent q-points for completeness.
"""

import numpy as np
import sys
from pathlib import Path

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).parent))

from phonproj.modes import PhononModes


def test_inequivalent_qpoints():
    """Test completeness using only inequivalent q-points."""

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
    N = int(np.round(np.linalg.det(supercell_matrix)))

    all_commensurate_displacements = modes.generate_all_commensurate_displacements(
        supercell_matrix, amplitude=1.0
    )

    print("=== Q-POINT EQUIVALENCE ANALYSIS ===")
    print("All q-points:")
    for i, qpoint in enumerate(qpoints_2x2x2):
        print(f"  Q{i}: {qpoint}")

    # For a cubic lattice, q-points are equivalent if they differ by reciprocal lattice vectors
    # or are related by symmetry operations

    # Let's test different subsets of q-points and see their completeness
    test_subsets = [
        ([0], "Gamma only"),
        ([0, 1], "Gamma + [0,0,0.5]"),
        ([0, 1, 2], "First 3 q-points"),
        ([0, 1, 2, 4], "4 q-points"),
        ([0, 2, 4, 6], "Alternating q-points"),
        ([0, 7], "Gamma + corner"),
        (list(range(8)), "All q-points"),
    ]

    supercell_masses = np.tile(modes.atomic_masses, N)

    # Create test vector
    np.random.seed(123)
    n_supercell_atoms = N * modes._n_atoms
    random_displacement = np.random.rand(n_supercell_atoms, 3)
    current_norm = modes.mass_weighted_norm(random_displacement, supercell_masses)
    normalized_displacement = random_displacement / current_norm

    print("\n=== COMPLETENESS TESTS FOR DIFFERENT Q-POINT SUBSETS ===")

    for q_indices, description in test_subsets:
        print(f"\n{description}: Q-points {q_indices}")

        # Count total modes
        total_modes = 0
        subset_displacements = {}
        for q_idx in q_indices:
            if q_idx in all_commensurate_displacements:
                subset_displacements[q_idx] = all_commensurate_displacements[q_idx]
                total_modes += all_commensurate_displacements[q_idx].shape[0]

        print(f"  Total modes: {total_modes}")

        # Test completeness
        sum_projections_squared = 0.0
        for q_index, displacements in subset_displacements.items():
            for i in range(displacements.shape[0]):
                projection = modes.mass_weighted_projection(
                    normalized_displacement, displacements[i], supercell_masses
                )
                sum_projections_squared += abs(projection) ** 2

        print(f"  Completeness sum: {sum_projections_squared:.6f}")
        print(f"  Deviation from 1.0: {abs(sum_projections_squared - 1.0):.6f}")

        # Check if this is close to 1.0
        if abs(sum_projections_squared - 1.0) < 0.1:
            print(f"  ✓ EXCELLENT completeness!")
        elif abs(sum_projections_squared - 1.0) < 0.5:
            print(f"  ✓ Good completeness")
        else:
            print(f"  ❌ Poor completeness")

    print("\n=== THEORETICAL ANALYSIS ===")
    print(
        "For a 2×2×2 supercell, the inequivalent q-points depend on crystal symmetry."
    )
    print("In the Monkhorst-Pack scheme:")
    print("  - [0,0,0] is always inequivalent (Gamma)")
    print("  - [0.5,0.5,0.5] might be equivalent to [0,0,0] under zone folding")
    print("  - Other q-points may be equivalent to each other")

    # Let's also test with fewer q-points systematically
    print("\n=== SYSTEMATIC REDUCTION TEST ===")

    # Try removing one q-point at a time and see the effect
    for remove_idx in range(8):
        remaining_indices = [i for i in range(8) if i != remove_idx]

        total_modes = 0
        subset_displacements = {}
        for q_idx in remaining_indices:
            if q_idx in all_commensurate_displacements:
                subset_displacements[q_idx] = all_commensurate_displacements[q_idx]
                total_modes += all_commensurate_displacements[q_idx].shape[0]

        sum_projections_squared = 0.0
        for q_index, displacements in subset_displacements.items():
            for i in range(displacements.shape[0]):
                projection = modes.mass_weighted_projection(
                    normalized_displacement, displacements[i], supercell_masses
                )
                sum_projections_squared += abs(projection) ** 2

        print(
            f"Remove Q{remove_idx} {qpoints_2x2x2[remove_idx]}: completeness = {sum_projections_squared:.6f} (modes: {total_modes})"
        )


if __name__ == "__main__":
    test_inequivalent_qpoints()
