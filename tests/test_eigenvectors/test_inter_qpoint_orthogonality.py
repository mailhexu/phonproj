#!/usr/bin/env python3
"""
Test inter-q-point orthogonality which was the original issue.
"""

import numpy as np
import sys

sys.path.insert(0, ".")

from phonproj.modes import PhononModes


def test_inter_qpoint_orthogonality():
    """Test orthogonality between different q-points - the original problem."""
    print("Testing inter-q-point orthogonality...")

    # Load test data with multiple q-points
    BATIO3_YAML_PATH = "data/BaTiO3_phonopy_params.yaml"

    # Use commensurate q-points for a 2x2x1 supercell
    qpoints = np.array(
        [
            [0.0, 0.0, 0.0],  # Gamma
            [0.5, 0.0, 0.0],  # X
            [0.0, 0.5, 0.0],  # Y
            [0.5, 0.5, 0.0],  # M
        ]
    )

    try:
        modes = PhononModes.from_phonopy_yaml(BATIO3_YAML_PATH, qpoints=qpoints)
        print(f"✓ Loaded {modes.n_qpoints} q-points with {modes.n_modes} modes each")
    except Exception as e:
        print(f"Error loading modes with multiple q-points: {e}")
        # Fall back to Gamma only
        qpoints = np.array([[0.0, 0.0, 0.0]])
        modes = PhononModes.from_phonopy_yaml(BATIO3_YAML_PATH, qpoints=qpoints)
        print(f"✓ Falling back to Gamma only: {modes.n_qpoints} q-points")

    supercell_matrix = np.array([[2, 0, 0], [0, 2, 0], [0, 0, 1]])
    print(f"✓ Using supercell matrix: {supercell_matrix.tolist()}")

    # Generate displacements for a few representative modes
    print("\nGenerating displacements...")
    test_cases = []

    # Add test cases for each q-point (first few modes)
    for q_idx in range(min(modes.n_qpoints, 4)):
        for mode_idx in range(min(3, modes.n_modes)):
            test_cases.append((q_idx, mode_idx))

    displacements = []
    labels = []

    for q_idx, mode_idx in test_cases:
        displacement = modes.generate_mode_displacement(
            q_index=q_idx,
            mode_index=mode_idx,
            supercell_matrix=supercell_matrix,
            amplitude=1.0,
        )
        displacements.append(displacement.flatten())
        labels.append(f"q{q_idx}_m{mode_idx}")
        print(f"  q={q_idx}, mode={mode_idx}: norm={np.linalg.norm(displacement):.6f}")

    # Test orthogonality matrix
    print("\nTesting orthogonality...")
    n_displacements = len(displacements)

    ortho_issues = 0
    max_cross_projection = 0.0

    for i in range(n_displacements):
        for j in range(i, n_displacements):
            dot_product = np.dot(displacements[i], displacements[j])

            qi, mi = test_cases[i]
            qj, mj = test_cases[j]

            if i == j:
                # Self projection
                print(f"  {labels[i]} self-projection: {dot_product:.6f}")
            else:
                print(f"  {labels[i]} - {labels[j]}: {dot_product:.6f}")

                # Check for problematic cross-projections
                if abs(dot_product) > 1e-6:
                    ortho_issues += 1
                    max_cross_projection = max(max_cross_projection, abs(dot_product))

                    if qi != qj:
                        print(f"    ⚠ INTER-Q-POINT non-orthogonality! q{qi} vs q{qj}")
                    else:
                        print(f"    ⚠ INTRA-Q-POINT non-orthogonality! (same q-point)")

    print(f"\nSummary:")
    print(f"  Total orthogonality issues: {ortho_issues}")
    print(f"  Maximum cross-projection: {max_cross_projection:.6f}")

    if ortho_issues == 0:
        print("✅ Perfect orthogonality achieved!")
        return True
    else:
        print("❌ Orthogonality issues remain")
        return False


if __name__ == "__main__":
    success = test_inter_qpoint_orthogonality()
    sys.exit(0 if success else 1)
