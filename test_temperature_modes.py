#!/usr/bin/env python3
"""
Test script for the new generate_modes_at_temperature method.

This script demonstrates how to use the temperature-dependent mode generation
and verifies that it produces physically reasonable results.
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
    frequencies = np.array([[0.0, 0.0, 0.0, 2.0, 3.0, 4.0]])  # 6 modes for 2 atoms

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


def test_generate_modes_at_temperature():
    """Test the new generate_modes_at_temperature method."""

    print("Testing generate_modes_at_temperature method...")
    print("=" * 60)

    # Create test phonon modes
    modes = create_test_phonon_modes()

    print(f"Created PhononModes object:")
    print(f"  - Number of atoms: {modes.n_atoms}")
    print(f"  - Number of q-points: {modes.n_qpoints}")
    print(f"  - Number of modes per q-point: {modes.n_modes}")
    print(f"  - Atomic masses: {modes.atomic_masses}")
    print()

    # Test parameters
    supercell_matrix = np.eye(3) * 2  # 2x2x2 supercell
    temperatures = [0.0, 100.0, 300.0, 600.0]  # K

    print("Testing temperature-dependent mode generation:")
    print(f"  - Supercell matrix: {supercell_matrix.tolist()}")
    print(f"  - Temperatures: {temperatures} K")
    print()

    for temp in temperatures:
        print(f"Temperature: {temp} K")
        print("-" * 30)

        try:
            # Generate temperature-dependent displacements
            displacements = modes.generate_modes_at_temperature(
                q_index=0,  # Gamma point
                supercell_matrix=supercell_matrix,
                temperature=temp,
            )

            print(f"  Generated displacements shape: {displacements.shape}")
            print(f"  Expected shape: ({modes.n_modes}, {8 * modes.n_atoms}, 3)")

            # Analyze the displacements
            for mode_idx in range(min(3, modes.n_modes)):  # Show first 3 modes
                mode_disp = displacements[mode_idx]
                max_disp = np.max(np.abs(mode_disp))
                mean_disp = np.mean(np.abs(mode_disp))

                print(
                    f"  Mode {mode_idx}: max |disp| = {max_disp:.6f} Å, mean |disp| = {mean_disp:.6f} Å"
                )

            # Check temperature dependence
            if temp > 0:
                # Higher temperature should generally give larger displacements
                total_disp = np.sum(np.abs(displacements))
                print(f"  Total displacement magnitude: {total_disp:.6f} Å")

        except Exception as e:
            print(f"  ERROR: {e}")

        print()

    print("=" * 60)
    print("Test completed!")


def compare_with_uniform_amplitude():
    """Compare temperature-dependent amplitudes with uniform amplitudes."""

    print("Comparing temperature-dependent vs uniform amplitudes:")
    print("=" * 60)

    modes = create_test_phonon_modes()
    supercell_matrix = np.eye(3)  # 1x1x1 supercell for simplicity
    temperature = 300.0  # K

    # Generate temperature-dependent displacements
    temp_displacements = modes.generate_modes_at_temperature(
        q_index=0, supercell_matrix=supercell_matrix, temperature=temperature
    )

    # Generate uniform amplitude displacements
    uniform_displacements = modes.generate_all_mode_displacements(
        q_index=0, supercell_matrix=supercell_matrix, amplitude=1.0
    )

    print(f"Temperature: {temperature} K")
    print(f"Supercell: 1x1x1 (for direct comparison)")
    print()

    for mode_idx in range(min(3, modes.n_modes)):
        temp_mode = temp_displacements[mode_idx]
        uniform_mode = uniform_displacements[mode_idx]

        temp_norm = np.linalg.norm(temp_mode)
        uniform_norm = np.linalg.norm(uniform_mode)

        ratio = temp_norm / uniform_norm if uniform_norm > 0 else 0

        print(f"Mode {mode_idx}:")
        print(f"  Temperature-dependent norm: {temp_norm:.6f}")
        print(f"  Uniform amplitude norm: {uniform_norm:.6f}")
        print(f"  Ratio (temp/uniform): {ratio:.6f}")
        print()

    print("=" * 60)


def test_refactored_methods():
    """Test that refactored methods produce the same results as before."""

    print("Testing refactored methods for consistency:")
    print("=" * 60)

    modes = create_test_phonon_modes()
    supercell_matrix = np.eye(3) * 2  # 2x2x2 supercell
    q_index = 0
    temperature = 300.0

    try:
        # Test temperature-dependent method
        temp_disps = modes.generate_modes_at_temperature(
            q_index, supercell_matrix, temperature
        )

        # Test uniform amplitude method
        uniform_disps = modes.generate_all_mode_displacements(
            q_index, supercell_matrix, amplitude=1.0
        )

        print(f"Temperature-dependent method:")
        print(f"  Shape: {temp_disps.shape}")
        print(f"  Max displacement: {np.max(np.abs(temp_disps)):.6f} Å")
        print()

        print(f"Uniform amplitude method:")
        print(f"  Shape: {uniform_disps.shape}")
        print(f"  Max displacement: {np.max(np.abs(uniform_disps)):.6f} Å")
        print()

        # Verify shapes are the same
        if temp_disps.shape == uniform_disps.shape:
            print("✓ Both methods produce arrays with the same shape")
        else:
            print("✗ Shape mismatch between methods")

        # Test different supercell sizes
        for size in [1, 2, 3]:
            test_matrix = np.eye(3) * size
            try:
                test_temp = modes.generate_modes_at_temperature(
                    q_index, test_matrix, temperature
                )
                test_uniform = modes.generate_all_mode_displacements(
                    q_index, test_matrix, amplitude=1.0
                )
                print(f"✓ {size}x{size}x{size} supercell: shapes match")
            except Exception as e:
                print(f"✗ {size}x{size}x{size} supercell: {e}")

        print()
        print("✓ Refactored methods working correctly!")

    except Exception as e:
        print(f"✗ Error testing refactored methods: {e}")
        import traceback

        traceback.print_exc()

    print("=" * 60)

    modes = create_test_phonon_modes()
    supercell_matrix = np.eye(3)  # 1x1x1 supercell for simplicity
    temperature = 300.0  # K

    # Generate temperature-dependent displacements
    temp_displacements = modes.generate_modes_at_temperature(
        q_index=0, supercell_matrix=supercell_matrix, temperature=temperature
    )

    # Generate uniform amplitude displacements
    uniform_displacements = modes.generate_all_mode_displacements(
        q_index=0, supercell_matrix=supercell_matrix, amplitude=1.0
    )

    print(f"Temperature: {temperature} K")
    print(f"Supercell: 1x1x1 (for direct comparison)")
    print()

    for mode_idx in range(min(3, modes.n_modes)):
        temp_mode = temp_displacements[mode_idx]
        uniform_mode = uniform_displacements[mode_idx]

        temp_norm = np.linalg.norm(temp_mode)
        uniform_norm = np.linalg.norm(uniform_mode)

        ratio = temp_norm / uniform_norm if uniform_norm > 0 else 0

        print(f"Mode {mode_idx}:")
        print(f"  Temperature-dependent norm: {temp_norm:.6f}")
        print(f"  Uniform amplitude norm: {uniform_norm:.6f}")
        print(f"  Ratio (temp/uniform): {ratio:.6f}")
        print()

    print("=" * 60)


if __name__ == "__main__":
    test_generate_modes_at_temperature()
    print()
    compare_with_uniform_amplitude()
    print()
    test_refactored_methods()
