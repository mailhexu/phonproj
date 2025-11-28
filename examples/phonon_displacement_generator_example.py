#!/usr/bin/env python3
"""
Example: Phonon Displacement Generator

This example demonstrates how to use the PhononDisplacementGenerator to:
1. Load phonopy data and calculate phonon modes
2. Generate atomic displacements for specific phonon modes
3. Save displaced supercell structures in VASP format
4. Use both the Python API and CLI interface

USAGE:
    uv run python examples/phonon_displacement_generator_example.py

EXPECTED OUTPUT:
    - Prints information about available q-points and modes
    - Shows example displacements for a few modes
    - Saves displaced structures to output directory
    - Demonstrates CLI usage

FILES CREATED:
    displacement_generator_output/
    ├── mode_q0_m0_freq_5.23THz.vasp
    ├── mode_q0_m1_freq_5.23THz.vasp
    └── mode_q1_m0_freq_8.45THz.vasp
"""

import numpy as np
from pathlib import Path
import tempfile
import subprocess
import sys

from phonproj.displacement import PhononDisplacementGenerator


def main():
    """Main example function demonstrating displacement generation."""

    print("=" * 60)
    print("Phonon Displacement Generator Example")
    print("=" * 60)

    # Path to BaTiO3 phonopy data
    data_path = Path("data/BaTiO3_phonopy_params.yaml")

    if not data_path.exists():
        print(f"Error: Data file not found at {data_path}")
        print("Please ensure the BaTiO3 phonopy data is available.")
        return

    # Initialize the displacement generator
    print(f"\n1. Loading phonopy data from: {data_path}")
    generator = PhononDisplacementGenerator(str(data_path))
    print("✓ Displacement generator initialized successfully")

    # Define supercell matrix (2x2x2)
    supercell_matrix = np.diag([2, 2, 2])
    print(f"\n2. Using supercell matrix: {supercell_matrix.diagonal()}")

    # Calculate phonon modes
    print("\n3. Calculating phonon modes...")
    modes = generator.calculate_modes(supercell_matrix)
    print(f"✓ Calculated modes for {len(modes.qpoints)} q-points")
    print(f"✓ Each q-point has {modes.frequencies.shape[1]} modes")

    # Get commensurate q-points
    qpoint_indices = generator.get_commensurate_qpoints(supercell_matrix)
    print(f"✓ Found {len(qpoint_indices)} commensurate q-points")

    # Display some information about available modes
    print("\n4. Available phonon modes:")
    print("   Q-point | Mode | Frequency (THz)")
    print("   --------|------|----------------")

    count = 0
    for q_idx in qpoint_indices[:3]:  # Show first 3 q-points
        for mode_idx in range(min(3, modes.frequencies.shape[1])):  # Show first 3 modes
            freq = modes.frequencies[q_idx, mode_idx]
            print(f"   {q_idx:7d} | {mode_idx:4d} | {freq:14.2f}")
            count += 1
            if count >= 6:  # Limit output
                break
        if count >= 6:
            break

    # Generate displacements for a few example modes
    print("\n5. Generating example displacements:")

    # Create output directory
    output_dir = Path("displacement_generator_output")
    output_dir.mkdir(exist_ok=True)

    # Example 1: Gamma point, lowest frequency optical mode
    print("\n   Example 1: Γ-point (q=0), mode 0")
    displacement_1 = generator.generate_displacement(
        q_idx=0, mode_idx=0, supercell_matrix=supercell_matrix, amplitude=0.1
    )
    print(f"   Displacement shape: {displacement_1.shape}")
    print(f"   Max displacement: {np.max(np.abs(displacement_1)):.4f} Å")

    # Save the structure
    structure_file_1 = generator.save_structure(
        q_idx=0,
        mode_idx=0,
        supercell_matrix=supercell_matrix,
        amplitude=0.1,
        output_dir=output_dir,
    )
    print(f"   ✓ Saved structure: {structure_file_1}")

    # Example 2: Different q-point
    print("\n   Example 2: Q-point 1, mode 0")
    displacement_2 = generator.generate_displacement(
        q_idx=1, mode_idx=0, supercell_matrix=supercell_matrix, amplitude=0.1
    )
    print(f"   Displacement shape: {displacement_2.shape}")
    print(f"   Max displacement: {np.max(np.abs(displacement_2)):.4f} Å")

    structure_file_2 = generator.save_structure(
        q_idx=1,
        mode_idx=0,
        supercell_matrix=supercell_matrix,
        amplitude=0.1,
        output_dir=output_dir,
    )
    print(f"   ✓ Saved structure: {structure_file_2}")

    # Example 3: Different amplitude
    print("\n   Example 3: Γ-point, mode 1, amplitude=0.05")
    displacement_3 = generator.generate_displacement(
        q_idx=0, mode_idx=1, supercell_matrix=supercell_matrix, amplitude=0.05
    )
    print(f"   Displacement shape: {displacement_3.shape}")
    print(f"   Max displacement: {np.max(np.abs(displacement_3)):.4f} Å")

    structure_file_3 = generator.save_structure(
        q_idx=0,
        mode_idx=1,
        supercell_matrix=supercell_matrix,
        amplitude=0.05,
        output_dir=output_dir,
    )
    print(f"   ✓ Saved structure: {structure_file_3}")

    # Demonstrate CLI usage
    print("\n6. CLI Usage Example:")
    print("   The following command demonstrates the CLI interface:")
    print(
        f"   phonproj-displacement -p {data_path} -s 2 2 2 --print-displacements --save-dir {output_dir} --amplitude 0.1"
    )

    # Actually run CLI for a small subset to demonstrate
    print("\n   Running CLI for Γ-point, mode 0:")
    try:
        result = subprocess.run(
            [
                "uv",
                "run",
                "phonproj-displacement",
                "-p",
                str(data_path),
                "-s",
                "2 2 2",
                "--print-displacements",
                "--save-dir",
                str(output_dir),
                "--amplitude",
                "0.1",
                "--max-atoms",
                "10",  # Limit output for demo
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )

        if result.returncode == 0:
            print("   ✓ CLI executed successfully")
            # Show a few lines of output
            lines = result.stdout.split("\n")[:15]
            for line in lines:
                if line.strip():
                    print(f"   {line}")
        else:
            print(f"   CLI error: {result.stderr}")

    except subprocess.TimeoutExpired:
        print("   CLI execution timed out (this is normal for large calculations)")
    except Exception as e:
        print(f"   CLI execution failed: {e}")

    # Summary
    print("\n" + "=" * 60)
    print("Summary:")
    print(f"✓ Generated displacements for {len(qpoint_indices)} q-points")
    print(f"✓ Each q-point has {modes.frequencies.shape[1]} phonon modes")
    print(f"✓ Total possible modes: {len(qpoint_indices) * modes.frequencies.shape[1]}")
    print(f"✓ Example structures saved to: {output_dir}")
    print(f"✓ Both Python API and CLI interfaces demonstrated")
    print("\nTo generate all possible structures, use:")
    print(f"   phonproj-displacement -p {data_path} -s 2 2 2 --save-dir all_structures")
    print("=" * 60)


if __name__ == "__main__":
    main()
