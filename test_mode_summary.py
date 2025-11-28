#!/usr/bin/env python3
"""
Test script for the new mode summary table functionality in PhononModes.

This script demonstrates how to use the new methods that provide
frequency and symmetry labeling similar to irreps_anaddb.py.
"""

import numpy as np
from ase import Atoms
from phonproj.modes import PhononModes


def create_test_phonon_modes():
    """Create a simple test PhononModes object for demonstration."""

    # Create a simple cubic unit cell with 2 atoms
    cell = np.eye(3) * 4.0  # 4 Å lattice parameter
    positions = [[0, 0, 0], [0.5, 0.5, 0.5]]  # Simple cubic structure
    symbols = ["A", "B"]
    masses = [50.0, 30.0]  # Atomic masses in amu

    primitive_cell = Atoms(symbols=symbols, positions=positions, cell=cell, pbc=True)

    # Create test data for Gamma point only
    qpoints = np.array([[0.0, 0.0, 0.0]])  # Gamma point

    # Create test frequencies (THz) - 3 acoustic + 3 optical for 2 atoms
    frequencies = np.array([[0.0, 0.0, 0.0, 2.5, 4.0, 6.0]])  # 6 modes for 2 atoms

    # Create test eigenvectors (complex, normalized)
    n_atoms = len(primitive_cell)
    n_modes = 3 * n_atoms  # 3N modes
    eigenvectors = np.random.random((1, n_modes, n_atoms * 3)) + 1j * np.random.random(
        (1, n_modes, n_atoms * 3)
    )

    # Normalize eigenvectors
    for mode_idx in range(n_modes):
        norm = np.linalg.norm(eigenvectors[0, mode_idx])
        if norm > 0:
            eigenvectors[0, mode_idx] /= norm

    return PhononModes(
        primitive_cell=primitive_cell,
        qpoints=qpoints,
        frequencies=frequencies,
        eigenvectors=eigenvectors,
        atomic_masses=np.array(masses),
        gauge="R",
    )


def test_mode_summary_table():
    """Test the new mode summary table functionality."""

    print("Testing mode summary table functionality:")
    print("=" * 60)

    # Create test phonon modes
    modes = create_test_phonon_modes()

    print(f"Created PhononModes object:")
    print(f"  - Number of atoms: {modes.n_atoms}")
    print(f"  - Number of q-points: {modes.n_qpoints}")
    print(f"  - Number of modes per q-point: {modes.n_modes}")
    print(f"  - Atomic masses: {modes.atomic_masses}")
    print()

    # Test mode summary table
    print("Mode Summary Table:")
    print("-" * 40)

    try:
        # Get summary table as data
        summary = modes.get_mode_summary_table(q_index=0)

        print(f"Generated summary with {len(summary)} modes:")
        print()

        # Print a few entries to show structure
        for i, row in enumerate(summary[:3]):  # Show first 3 modes
            print(f"Mode {row['band_index']}:")
            print(f"  Frequency (THz): {row['frequency_thz']:.4f}")
            print(f"  Frequency (cm⁻¹): {row['frequency_cm1']:.2f}")
            print(f"  Label: {row['label']}")
            print(f"  IR active: {row['is_ir_active']}")
            print(f"  Raman active: {row['is_raman_active']}")
            print()

        print("Formatted Table Output:")
        print("-" * 40)

        # Print formatted table
        table_output = modes.print_mode_summary_table(q_index=0, include_header=True)
        print(table_output)

        print()
        print("✓ Mode summary table functionality working correctly!")

    except Exception as e:
        print(f"✗ Error testing mode summary table: {e}")
        import traceback

        traceback.print_exc()

    print("=" * 60)


def test_multiple_qpoints():
    """Test summary table for multiple q-points."""

    print("Testing multiple q-points:")
    print("=" * 60)

    # Create a PhononModes object with multiple q-points
    modes = create_test_phonon_modes()

    # Add more q-points by creating a new object
    qpoints = np.array(
        [
            [0.0, 0.0, 0.0],  # Gamma
            [0.5, 0.0, 0.0],  # X point
            [0.0, 0.5, 0.0],  # Y point
        ]
    )

    frequencies = np.array(
        [
            [0.0, 0.0, 0.0, 2.5, 4.0, 6.0],  # Gamma point
            [1.0, 1.2, 1.5, 3.0, 4.5, 5.5],  # X point
            [0.8, 1.1, 1.3, 2.8, 4.2, 5.8],  # Y point
        ]
    )

    n_qpoints, n_modes = len(qpoints), len(frequencies[0])
    eigenvectors = np.random.random(
        (n_qpoints, n_modes, modes.n_atoms * 3)
    ) + 1j * np.random.random((n_qpoints, n_modes, modes.n_atoms * 3))

    # Normalize eigenvectors
    for q_idx in range(n_qpoints):
        for mode_idx in range(n_modes):
            norm = np.linalg.norm(eigenvectors[q_idx, mode_idx])
            if norm > 0:
                eigenvectors[q_idx, mode_idx] /= norm

    # Create new PhononModes object with multiple q-points
    multi_q_modes = PhononModes(
        primitive_cell=modes.primitive_cell,
        qpoints=qpoints,
        frequencies=frequencies,
        eigenvectors=eigenvectors,
        atomic_masses=modes.atomic_masses,
        gauge="R",
    )

    print(f"Created PhononModes with {multi_q_modes.n_qpoints} q-points:")

    # Test summary for each q-point
    for q_idx in range(multi_q_modes.n_qpoints):
        print(f"\n--- Q-point {q_idx} ---")
        try:
            table_output = multi_q_modes.print_mode_summary_table(
                q_index=q_idx, include_header=True
            )
            print(table_output)
        except Exception as e:
            print(f"Error for q-point {q_idx}: {e}")

    print()
    print("✓ Multiple q-point testing completed!")
    print("=" * 60)


def test_comparison_with_irreps_format():
    """Test that the output format matches irreps_anaddb.py format."""

    print("Testing format compatibility with irreps_anaddb.py:")
    print("=" * 60)

    modes = create_test_phonon_modes()

    # Get the formatted output
    output = modes.print_mode_summary_table(q_index=0, include_header=True)

    print("Generated output:")
    print(output)
    print()

    # Check format elements
    lines = output.split("\n")

    # Check header
    header_line = None
    for line in lines:
        if line.startswith("# qx"):
            header_line = line
            break

    if header_line:
        print("✓ Header line found:", header_line)
        expected_columns = [
            "qx",
            "qy",
            "qz",
            "band",
            "freq(THz)",
            "freq(cm-1)",
            "label",
            "IR",
            "Raman",
        ]
        actual_columns = header_line.split()[1:]  # Skip '#'
        print(f"  Expected columns: {expected_columns}")
        print(f"  Actual columns: {actual_columns}")
        print()

    # Check data lines
    data_lines = [
        line
        for line in lines
        if not line.startswith("#")
        and not line.startswith("q-point:")
        and not line.startswith("Point group:")
        and line.strip()
    ]

    if data_lines:
        print(f"✓ Found {len(data_lines)} data lines")

        # Parse first data line to check format
        first_data = data_lines[0].split()
        if len(first_data) >= 9:
            print(f"  First data line parsed successfully:")
            print(f"    qx: {first_data[0]}")
            print(f"    qy: {first_data[1]}")
            print(f"    qz: {first_data[2]}")
            print(f"    band: {first_data[3]}")
            print(f"    freq(THz): {first_data[4]}")
            print(f"    freq(cm-1): {first_data[5]}")
            print(f"    label: {first_data[6]}")
            print(f"    IR: {first_data[7]}")
            print(f"    Raman: {first_data[8]}")

    print()
    print("✓ Format compatibility testing completed!")
    print("=" * 60)


if __name__ == "__main__":
    test_mode_summary_table()
    print()
    test_multiple_qpoints()
    print()
    test_comparison_with_irreps_format()
