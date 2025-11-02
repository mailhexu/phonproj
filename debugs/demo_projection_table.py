#!/usr/bin/env python3
"""
Demonstration of the projection table functionality.

This script shows how to use the new print_projection_table() method to analyze
how a displacement decomposes across all commensurate q-point modes.
"""

import numpy as np
from pathlib import Path
from phonproj.modes import PhononModes

# Test data paths
BATIO3_YAML_PATH = Path(__file__).parent / "data" / "BaTiO3_phonopy_params.yaml"


def demo_projection_table_2x2x1():
    """Demonstrate projection table for 2x2x1 supercell with BaTiO3."""
    print("=" * 80)
    print("DEMO: Projection Table Analysis for 2x2x1 Supercell")
    print("=" * 80)

    # Load BaTiO3 data with 2x2x1 commensurate q-points
    qpoints_2x2x1 = []
    for i in range(2):
        for j in range(2):
            for k in range(1):
                qpoints_2x2x1.append([i / 2.0, j / 2.0, k / 1.0])
    qpoints_2x2x1 = np.array(qpoints_2x2x1)

    print(f"Loading BaTiO3 with {len(qpoints_2x2x1)} q-points:")
    for i, q in enumerate(qpoints_2x2x1):
        print(f"  Q{i}: [{q[0]:.1f}, {q[1]:.1f}, {q[2]:.1f}]")

    modes = PhononModes.from_phonopy_yaml(str(BATIO3_YAML_PATH), qpoints_2x2x1)
    supercell_matrix = np.array([[2, 0, 0], [0, 2, 0], [0, 0, 1]])

    print(f"\nPrimitive cell: {modes.n_atoms} atoms")
    print(f"Supercell: {4 * modes.n_atoms} atoms (2×2×1)")

    # Example 1: Random displacement
    print("\n" + "+" * 60)
    print("EXAMPLE 1: Random Displacement Analysis")
    print("+" * 60)

    np.random.seed(42)
    n_supercell_atoms = 4 * modes.n_atoms
    random_displacement = np.random.rand(n_supercell_atoms, 3)

    print(f"Generated random displacement with {n_supercell_atoms} atoms")
    print("Analyzing projection onto all commensurate modes...")

    result1 = modes.print_projection_table(
        random_displacement,
        supercell_matrix,
        normalize_displacement=True,
        max_modes_per_qpoint=5,  # Show only first 5 modes per q-point for clarity
        show_frequencies=True,
        precision=6,
    )

    # Example 2: Single mode displacement
    print("\n" + "+" * 60)
    print("EXAMPLE 2: Single Mode Displacement Analysis")
    print("+" * 60)

    # Generate displacement for a specific mode (Gamma point, mode 5)
    gamma_index = 0  # First q-point should be Gamma
    mode_index = 5
    single_mode_displacement = modes.generate_mode_displacement(
        gamma_index, mode_index, supercell_matrix, amplitude=1.0
    )

    print(f"Generated displacement for Q-point {gamma_index}, mode {mode_index}")
    print(f"Frequency: {modes.frequencies[gamma_index, mode_index]:.3f} THz")
    print("This should project almost entirely onto the source mode...")

    result2 = modes.print_projection_table(
        single_mode_displacement,
        supercell_matrix,
        normalize_displacement=True,  # Let the method handle proper normalization
        max_modes_per_qpoint=8,
        show_frequencies=True,
        precision=8,
    )

    # Example 3: Superposition of two modes
    print("\n" + "+" * 60)
    print("EXAMPLE 3: Superposition of Two Modes")
    print("+" * 60)

    # Create superposition of two modes
    mode1_disp = modes.generate_mode_displacement(0, 3, supercell_matrix, amplitude=0.6)
    mode2_disp = modes.generate_mode_displacement(
        1,
        7,
        supercell_matrix,
        amplitude=0.8,  # Different q-point
    )

    # Combine them
    superposition = mode1_disp + mode2_disp

    print(f"Created superposition:")
    print(f"  0.6 × (Q0, mode 3) + 0.8 × (Q1, mode 7)")
    print(
        f"  Frequencies: {modes.frequencies[0, 3]:.3f} THz + {modes.frequencies[1, 7]:.3f} THz"
    )
    print("Should see dominant projections on these two modes...")

    result3 = modes.print_projection_table(
        superposition,
        supercell_matrix,
        normalize_displacement=True,  # Let the method handle proper normalization
        max_modes_per_qpoint=10,
        show_frequencies=True,
        precision=6,
    )

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY OF RESULTS")
    print("=" * 80)

    print(f"Example 1 (Random): Total sum = {result1.get('total_sum', 'N/A'):.6f}")
    print(f"Example 2 (Single): Total sum = {result2.get('total_sum', 'N/A'):.6f}")
    print(
        f"Example 3 (Superposition): Total sum = {result3.get('total_sum', 'N/A'):.6f}"
    )

    print(
        f"\nAll examples used {result1.get('n_qpoints', 'N/A')} commensurate q-points"
    )
    print(f"Total modes available: {result1.get('n_total_modes', 'N/A')}")

    print("\nKey observations:")
    print("• Random displacement: Should show roughly equal distribution across modes")
    print("• Single mode: Should show ~1.0 projection on source mode, ~0.0 elsewhere")
    print("• Superposition: Should show projections matching input amplitudes²")
    print("• All cases: Total sum ≈ 1.0 confirms completeness of orthonormal basis")


def demo_projection_table_small():
    """Quick demo with fewer q-points for cleaner output."""
    print("\n" + "=" * 80)
    print("MINI DEMO: Projection Table with 1x1x2 Supercell")
    print("=" * 80)

    # Use 1x1x2 supercell (only 2 q-points)
    qpoints_1x1x2 = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.5]])
    modes = PhononModes.from_phonopy_yaml(str(BATIO3_YAML_PATH), qpoints_1x1x2)
    supercell_matrix = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 2]])

    print(f"Using 2 q-points: Gamma and [0, 0, 0.5]")
    print(f"Supercell: {2 * modes.n_atoms} atoms (1×1×2)")

    # Single mode displacement
    mode_displacement = modes.generate_mode_displacement(
        1,
        8,
        supercell_matrix,
        amplitude=1.0,  # Non-Gamma point, mode 8
    )

    print(f"\nGenerated displacement for Q-point 1, mode 8")
    print(f"Frequency: {modes.frequencies[1, 8]:.3f} THz")

    result = modes.print_projection_table(
        mode_displacement,
        supercell_matrix,
        normalize_displacement=True,  # Let the method handle proper normalization
        show_frequencies=True,
        precision=8,
    )

    print(f"\nResult: Total sum = {result.get('total_sum', 'N/A'):.8f}")
    print("This demonstrates perfect projection onto the source mode!")


if __name__ == "__main__":
    try:
        # Run the mini demo first for a quick overview
        demo_projection_table_small()

        # Run the full demo automatically
        print("\n" + "=" * 60)
        print("Running full demo with more examples...")
        demo_projection_table_2x2x1()

    except FileNotFoundError:
        print(f"Error: Could not find {BATIO3_YAML_PATH}")
        print("Please ensure the BaTiO3 data file is available.")
    except Exception as e:
        print(f"Error running demo: {e}")
        import traceback

        traceback.print_exc()
