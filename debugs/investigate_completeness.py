#!/usr/bin/env python3

import numpy as np
from pathlib import Path
from phonproj.modes import PhononModes


def investigate_completeness():
    """Investigate the completeness calculation in detail."""

    # Test data path
    BATIO3_YAML_PATH = Path("data/BaTiO3_phonopy_params.yaml")

    # Load BaTiO3 data for 2x2x2 supercell
    qpoints_2x2x2 = []
    for i in range(2):
        for j in range(2):
            for k in range(2):
                qpoints_2x2x2.append([i / 2.0, j / 2.0, k / 2.0])
    qpoints_2x2x2 = np.array(qpoints_2x2x2)

    modes = PhononModes.from_phonopy_yaml(str(BATIO3_YAML_PATH), qpoints_2x2x2)

    # Use 2x2x2 supercell
    supercell_matrix = np.eye(3, dtype=int) * 2

    print("=== INVESTIGATING COMPLETENESS CALCULATION ===")
    print(f"Number of atoms in primitive cell: {modes._n_atoms}")
    print(f"Number of modes per q-point: {modes.n_modes}")
    print(f"Supercell: 2x2x2, N = 8 primitive cells")
    print(f"Number of atoms in supercell: {8 * modes._n_atoms}")
    print(f"Degrees of freedom in supercell: {8 * modes._n_atoms * 3}")

    # Generate displacements for all commensurate q-points
    all_commensurate_displacements = modes.generate_all_commensurate_displacements(
        supercell_matrix, amplitude=1.0
    )

    print(
        f"\nNumber of commensurate q-points found: {len(all_commensurate_displacements)}"
    )

    # Count total number of eigenmodes
    total_eigenmodes = 0
    non_zero_eigenmodes = 0

    for q_index, displacements in all_commensurate_displacements.items():
        q_eigenmodes = 0
        q_nonzero = 0
        for i in range(displacements.shape[0]):
            norm = modes.mass_weighted_norm(displacements[i])
            total_eigenmodes += 1
            if norm > 1e-10:
                non_zero_eigenmodes += 1
                q_nonzero += 1
            q_eigenmodes += 1
        print(
            f"Q-point {q_index}: {q_eigenmodes} total modes, {q_nonzero} non-zero modes"
        )

    print(f"\nTotal eigenmodes: {total_eigenmodes}")
    print(f"Non-zero eigenmodes: {non_zero_eigenmodes}")
    print(f"Expected total modes for complete basis: {8 * modes._n_atoms * 3}")

    # Check if we have the right number of modes for completeness
    expected_modes = 8 * modes._n_atoms * 3  # 3N degrees of freedom in supercell
    if non_zero_eigenmodes != expected_modes:
        print(
            f"⚠️  WARNING: Expected {expected_modes} modes but have {non_zero_eigenmodes}"
        )
    else:
        print(f"✓ Correct number of modes for complete basis")

    # Create random displacement and test completeness step by step
    print(f"\n=== STEP-BY-STEP COMPLETENESS TEST ===")

    np.random.seed(123)
    n_supercell_atoms = 8 * modes._n_atoms
    random_displacement = np.random.rand(n_supercell_atoms, 3)

    # Normalize with mass-weighted norm 1
    supercell_masses = np.tile(modes.atomic_masses, 8)
    current_norm = modes.mass_weighted_norm(random_displacement, supercell_masses)
    normalized_displacement = random_displacement / current_norm

    print(
        f"Random displacement normalized to norm: {modes.mass_weighted_norm(normalized_displacement, supercell_masses):.10f}"
    )

    # Project onto eigenmodes one q-point at a time
    total_sum = 0.0

    for q_index, displacements in all_commensurate_displacements.items():
        q_sum = 0.0
        q_contributions = []

        for i in range(displacements.shape[0]):
            norm = modes.mass_weighted_norm(displacements[i])
            if norm > 1e-10:  # Only count non-zero modes
                projection = modes.mass_weighted_projection(
                    normalized_displacement, displacements[i], supercell_masses
                )
                contribution = abs(projection) ** 2
                q_sum += contribution
                q_contributions.append((i, norm, abs(projection), contribution))

        print(f"\nQ-point {q_index}: {len(q_contributions)} contributing modes")
        print(f"  Sum for this q-point: {q_sum:.6f}")

        # Show a few largest contributions
        q_contributions.sort(key=lambda x: x[3], reverse=True)
        for j, (mode_idx, norm, proj_abs, contrib) in enumerate(q_contributions[:3]):
            print(
                f"    Mode {mode_idx}: norm={norm:.6f}, |proj|={proj_abs:.6f}, contrib={contrib:.6f}"
            )

        total_sum += q_sum

    print(f"\nFinal completeness sum: {total_sum:.6f}")
    print(f"Expected for perfect orthonormal basis: 1.0")
    print(f"Ratio: {total_sum:.2f}x")

    # Test orthogonality between q-points
    print(f"\n=== TESTING INTER-Q-POINT ORTHOGONALITY ===")

    q_indices = list(all_commensurate_displacements.keys())
    max_inter_q_projection = 0.0

    for i, q1 in enumerate(q_indices):
        for j, q2 in enumerate(q_indices):
            if i < j:  # Only check each pair once
                disps1 = all_commensurate_displacements[q1]
                disps2 = all_commensurate_displacements[q2]

                for m1 in range(disps1.shape[0]):
                    for m2 in range(disps2.shape[0]):
                        norm1 = modes.mass_weighted_norm(disps1[m1])
                        norm2 = modes.mass_weighted_norm(disps2[m2])

                        if norm1 > 1e-10 and norm2 > 1e-10:
                            projection = abs(
                                modes.mass_weighted_projection(disps1[m1], disps2[m2])
                            )
                            max_inter_q_projection = max(
                                max_inter_q_projection, projection
                            )

    print(f"Maximum inter-q-point projection: {max_inter_q_projection:.10f}")

    if max_inter_q_projection < 1e-6:
        print("✓ Perfect inter-q-point orthogonality confirmed")
    else:
        print("⚠️  Inter-q-point orthogonality issues remain")


if __name__ == "__main__":
    investigate_completeness()
