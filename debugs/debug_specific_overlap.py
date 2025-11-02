#!/usr/bin/env python3
"""
Focused debug script to find which specific mode pairs have overlap = 1.0
"""

import numpy as np
from pathlib import Path
from phonproj.modes import PhononModes

# Test data path
BATIO3_YAML_PATH = Path(__file__).parent / "data" / "BaTiO3_phonopy_params.yaml"


def debug_specific_overlap():
    print("=== Finding Specific Mode Pairs with Overlap = 1.0 ===")

    # Generate all required q-points for 16x1x1 supercell
    qpoints_16x1x1 = []
    for i in range(16):
        qpoints_16x1x1.append([i / 16.0, 0.0, 0.0])
    qpoints_16x1x1 = np.array(qpoints_16x1x1)

    modes = PhononModes.from_phonopy_yaml(str(BATIO3_YAML_PATH), qpoints_16x1x1)

    # Use 16x1x1 supercell
    supercell_matrix = np.array([[16, 0, 0], [0, 1, 0], [0, 0, 1]])
    N = 16

    # Generate displacements for all commensurate q-points
    all_commensurate_displacements = modes.generate_all_commensurate_displacements(
        supercell_matrix, amplitude=1.0
    )

    # Collect all displacements exactly like the test does
    displacement_list = []
    qpoint_labels = []

    for q_index, displacements in all_commensurate_displacements.items():
        for mode_idx in range(displacements.shape[0]):
            displacement_list.append(displacements[mode_idx])
            qpoint_labels.append((q_index, mode_idx))

    # Check all pairs exactly like the test does
    n_total_modes = len(displacement_list)
    supercell_masses = np.tile(modes.atomic_masses, N)

    print(f"Total modes to check: {n_total_modes}")
    print(f"Total pairs to check: {n_total_modes * (n_total_modes - 1) // 2}")

    overlap_1_pairs = []
    max_overlap = 0.0

    for i in range(n_total_modes):
        for j in range(i + 1, n_total_modes):
            q_i, mode_i = qpoint_labels[i]
            q_j, mode_j = qpoint_labels[j]

            # Only check modes from different q-points like the test
            if q_i != q_j:
                projection = modes.mass_weighted_projection(
                    displacement_list[i], displacement_list[j], supercell_masses
                )
                overlap = abs(projection)
                max_overlap = max(max_overlap, overlap)

                # Find the pairs with overlap = 1.0
                if overlap > 0.99:  # Close to 1.0
                    overlap_1_pairs.append((i, j, q_i, mode_i, q_j, mode_j, overlap))

                # Report a few high overlaps for debugging
                if len(overlap_1_pairs) < 5 and overlap > 0.1:
                    print(
                        f"High overlap: Q{q_i} mode {mode_i} vs Q{q_j} mode {mode_j}: {overlap:.6f}"
                    )

    print(f"\nMax overlap found: {max_overlap:.6f}")
    print(f"Number of pairs with overlap > 0.99: {len(overlap_1_pairs)}")

    if overlap_1_pairs:
        print(f"\nFirst few high overlap pairs:")
        for i, (idx_i, idx_j, q_i, mode_i, q_j, mode_j, overlap) in enumerate(
            overlap_1_pairs[:10]
        ):
            print(
                f"  {i + 1}: Q{q_i} mode {mode_i} vs Q{q_j} mode {mode_j}: overlap = {overlap:.6f}"
            )

            # Let's check if these are actually the same displacement
            disp_i = displacement_list[idx_i]
            disp_j = displacement_list[idx_j]
            diff_norm = np.linalg.norm(disp_i - disp_j)
            print(f"      Displacement difference norm: {diff_norm:.2e}")

            # Check if they have the same norm
            norm_i = modes.mass_weighted_norm(disp_i, supercell_masses)
            norm_j = modes.mass_weighted_norm(disp_j, supercell_masses)
            print(f"      Norms: {norm_i:.6f} vs {norm_j:.6f}")

            # Check the raw displacement values
            print(f"      First atom of disp_i: {disp_i[0]}")
            print(f"      First atom of disp_j: {disp_j[0]}")
            print()

            if i >= 2:  # Limit output
                break

    # Let's also check if there are modes within the same q-point that are problematic
    print(f"\n=== Checking intra-q-point orthogonality ===")
    intra_violations = 0
    for q_index, displacements in all_commensurate_displacements.items():
        n_modes = displacements.shape[0]
        for i in range(n_modes):
            for j in range(i + 1, n_modes):
                projection = modes.mass_weighted_projection(
                    displacements[i], displacements[j], supercell_masses
                )
                overlap = abs(projection)
                if overlap > 1e-6:
                    intra_violations += 1
                    if intra_violations <= 3:  # Show first few
                        print(
                            f"  Q{q_index} mode {i} vs mode {j}: overlap = {overlap:.2e}"
                        )

    print(f"Intra-q-point orthogonality violations: {intra_violations}")


if __name__ == "__main__":
    debug_specific_overlap()
