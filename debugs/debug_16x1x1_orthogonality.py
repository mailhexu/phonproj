#!/usr/bin/env python3
"""
Debug script for investigating the 16x1x1 supercell orthogonality failure.

The issue: modes from different q-points have overlap = 1.0 instead of being orthogonal.
This suggests the modes are identical rather than properly distinct.
"""

import numpy as np
from pathlib import Path
from phonproj.modes import PhononModes

# Test data path
BATIO3_YAML_PATH = Path(__file__).parent / "data" / "BaTiO3_phonopy_params.yaml"


def debug_16x1x1_orthogonality():
    print("=== Debug 16x1x1 Supercell Orthogonality Issue ===")

    # Generate all required q-points for 16x1x1 supercell
    qpoints_16x1x1 = []
    for i in range(16):
        qpoints_16x1x1.append([i / 16.0, 0.0, 0.0])
    qpoints_16x1x1 = np.array(qpoints_16x1x1)

    print(f"Generated {len(qpoints_16x1x1)} q-points:")
    for i, qpt in enumerate(qpoints_16x1x1):
        print(f"  q{i}: [{qpt[0]:.3f}, {qpt[1]:.3f}, {qpt[2]:.3f}]")

    # Load the modes
    print(f"\nLoading BaTiO3 with {len(qpoints_16x1x1)} q-points...")
    modes = PhononModes.from_phonopy_yaml(str(BATIO3_YAML_PATH), qpoints_16x1x1)

    # Check if PHONOPY actually calculated all q-points
    print(f"\nPhonopy calculated q-points:")
    print(f"  Total: {len(modes.qpoints)}")
    for i, qpt in enumerate(modes.qpoints):
        print(f"  q{i}: [{qpt[0]:.3f}, {qpt[1]:.3f}, {qpt[2]:.3f}]")

    # Use 16x1x1 supercell
    supercell_matrix = np.array([[16, 0, 0], [0, 1, 0], [0, 0, 1]])
    N = 16  # Number of primitive cells

    print(f"\nSupercell matrix:\n{supercell_matrix}")
    print(f"Number of primitive cells: {N}")

    # Check commensurate q-points
    commensurate_qpoints = modes.get_commensurate_qpoints(supercell_matrix)
    print(f"\nCommensurate q-points found: {len(commensurate_qpoints)}")
    for i, qpt_idx in enumerate(commensurate_qpoints):
        qpt = modes.qpoints[qpt_idx]
        print(f"  Index {qpt_idx}: [{qpt[0]:.3f}, {qpt[1]:.3f}, {qpt[2]:.3f}]")

    # Generate displacements for all commensurate q-points
    print(f"\nGenerating displacements...")
    all_commensurate_displacements = modes.generate_all_commensurate_displacements(
        supercell_matrix, amplitude=1.0
    )

    print(f"Generated displacements for {len(all_commensurate_displacements)} q-points")

    # Look at the first few displacements from different q-points
    print(f"\n=== Checking displacement differences ===")

    displacement_list = []
    qpoint_labels = []

    for q_index, displacements in all_commensurate_displacements.items():
        print(f"\nQ-point index {q_index}:")
        print(f"  Q-point: {modes.qpoints[q_index]}")
        print(f"  Number of modes: {displacements.shape[0]}")
        print(f"  Displacement shape: {displacements.shape}")

        # Store first mode from each q-point for comparison
        if len(displacement_list) < 4:  # Only store first few for debugging
            displacement_list.append(displacements[0])  # First mode
            qpoint_labels.append((q_index, 0))

    # Compare displacements between different q-points
    print(f"\n=== Comparing first mode from different q-points ===")
    supercell_masses = np.tile(modes.atomic_masses, N)

    for i in range(min(4, len(displacement_list))):
        for j in range(i + 1, min(4, len(displacement_list))):
            q_i, mode_i = qpoint_labels[i]
            q_j, mode_j = qpoint_labels[j]

            disp_i = displacement_list[i]
            disp_j = displacement_list[j]

            # Check if displacements are identical
            diff_norm = np.linalg.norm(disp_i - disp_j)

            # Check mass-weighted projection
            projection = modes.mass_weighted_projection(
                disp_i, disp_j, supercell_masses
            )

            print(f"\nQ{q_i} mode {mode_i} vs Q{q_j} mode {mode_j}:")
            print(f"  Q-points: {modes.qpoints[q_i]} vs {modes.qpoints[q_j]}")
            print(f"  Displacement difference norm: {diff_norm:.2e}")
            print(f"  Mass-weighted projection: {abs(projection):.2e}")

            # Check if they're actually the same displacement
            if diff_norm < 1e-10:
                print(f"  *** IDENTICAL DISPLACEMENTS! ***")

            # Show first few values for visual inspection
            print(f"  First 3 atoms of disp_i: {disp_i[:3].flatten()}")
            print(f"  First 3 atoms of disp_j: {disp_j[:3].flatten()}")

    print(f"\n=== Checking eigenvectors directly ===")

    # Let's look at the raw eigenvectors before transformation
    for i, q_idx in enumerate(list(all_commensurate_displacements.keys())[:4]):
        print(f"\nQ-point {q_idx} ({modes.qpoints[q_idx]}):")

        # Get the raw eigenvectors
        eigenvectors_q = modes.eigenvectors[q_idx]  # Shape: (n_modes, n_atoms, 3)
        frequencies_q = modes.frequencies[q_idx]  # Shape: (n_modes,)

        print(f"  Eigenvectors shape: {eigenvectors_q.shape}")
        print(f"  Frequencies shape: {frequencies_q.shape}")
        print(f"  First frequency: {frequencies_q[0]:.6f}")
        print(f"  First eigenvector (first atom): {eigenvectors_q[0, 0]}")

        # Check if eigenvectors are identical across q-points
        if i > 0:
            prev_q_idx = list(all_commensurate_displacements.keys())[0]
            prev_eigenvectors = modes.eigenvectors[prev_q_idx]

            # Compare first mode
            diff = np.linalg.norm(eigenvectors_q[0] - prev_eigenvectors[0])
            print(f"  Difference from Q0 first mode: {diff:.2e}")

            if diff < 1e-10:
                print(f"  *** IDENTICAL EIGENVECTORS! ***")


if __name__ == "__main__":
    debug_16x1x1_orthogonality()
