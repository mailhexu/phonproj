"""
Enhanced Structure Mapping Example

This example demonstrates the enhanced structure mapping functionality introduced in Step 12.
It shows how to use the new PBC-aware distance calculations, origin alignment,
shift optimization, and detailed output generation.

The example uses test data from the yajun dataset to demonstrate real-world usage.
"""

import os
import numpy as np
from ase import Atoms
from ase.io import read

from phonproj.core.structure_analysis import (
    calculate_pbc_distance,
    find_closest_to_origin,
    shift_to_origin,
    create_enhanced_atom_mapping,
    MappingAnalyzer,
)


def demonstrate_pbc_distance_calculation():
    """
    Demonstrate PBC-aware distance calculation.
    """
    print("=" * 60)
    print("PBC Distance Calculation Demonstration")
    print("=" * 60)

    # Create a simple cubic cell
    cell = np.eye(3) * 10.0  # 10Å cubic cell

    # Two positions that are close across periodic boundary
    pos1 = np.array([0.5, 0.5, 0.5])
    pos2 = np.array([9.5, 9.5, 9.5])

    # Calculate distance with PBC consideration
    pbc_distance = calculate_pbc_distance(pos1, pos2, cell)
    direct_distance = np.linalg.norm(pos2 - pos1)

    print(f"Position 1: [{pos1[0]:.1f}, {pos1[1]:.1f}, {pos1[2]:.1f}]")
    print(f"Position 2: [{pos2[0]:.1f}, {pos2[1]:.1f}, {pos2[2]:.1f}]")
    print(f"Direct distance: {direct_distance:.3f} Å")
    print(f"PBC distance: {pbc_distance:.3f} Å")
    print(f"PBC reduction: {direct_distance / pbc_distance:.1f}x shorter")
    print()


def demonstrate_origin_alignment():
    """
    Demonstrate origin alignment functionality.
    """
    print("=" * 60)
    print("Origin Alignment Demonstration")
    print("=" * 60)

    # Create a structure with atoms at various positions
    positions = np.array(
        [
            [8.5, 8.5, 8.5],  # Closest to origin across PBC
            [2.0, 2.0, 2.0],  # Middle distance
            [5.0, 5.0, 5.0],  # Farthest from origin
        ]
    )
    structure = Atoms(
        symbols=["H", "He", "Li"], positions=positions, cell=np.eye(3) * 10.0, pbc=True
    )

    # Find closest atom to origin
    closest_idx, distance, closest_pos = find_closest_to_origin(structure)

    print(f"Structure has {len(structure)} atoms")
    print(f"Closest atom to origin: index {closest_idx}")
    print(f"Distance to origin: {distance:.3f} Å")
    print(
        f"Closest atom position: [{closest_pos[0]:.1f}, {closest_pos[1]:.1f}, {closest_pos[2]:.1f}]"
    )

    # Shift structure to place closest atom at origin
    shifted = shift_to_origin(structure, closest_idx)
    new_pos = shifted.get_positions()[closest_idx]

    print(
        f"After shifting, closest atom position: [{new_pos[0]:.3f}, {new_pos[1]:.3f}, {new_pos[2]:.3f}]"
    )
    print()


def demonstrate_enhanced_mapping():
    """
    Demonstrate enhanced atom mapping with real data.
    """
    print("=" * 60)
    print("Enhanced Atom Mapping Demonstration")
    print("=" * 60)

    # Try to load test data
    data_dir = "data/yajundata"
    ref_file = os.path.join(data_dir, "ref.vasp")

    if not os.path.exists(ref_file):
        print(f"Test data file {ref_file} not found.")
        print("Creating synthetic example instead...")
        demonstrate_synthetic_mapping()
        return

    # Load reference structure
    print(f"Loading reference structure from {ref_file}")
    ref_structure = read(ref_file)
    if isinstance(ref_structure, list):
        ref_structure = ref_structure[0]

    print(f"Reference structure: {len(ref_structure)} atoms")
    print(f"Cell vectors: {ref_structure.get_cell().array}")

    # Create a test structure by shuffling and translating
    print("\nCreating test structure (shuffled + translated)...")

    # Shuffle atoms
    shuffled_indices = np.random.permutation(len(ref_structure))
    shuffled_positions = ref_structure.get_positions()[shuffled_indices]
    shuffled_species = [
        ref_structure.get_chemical_symbols()[i] for i in shuffled_indices
    ]

    # Apply translation
    translation = np.array([0.5, 0.3, 0.7])  # Translation in scaled coordinates
    cell = ref_structure.get_cell()
    scaled_translation = translation @ cell
    translated_positions = shuffled_positions + scaled_translation

    # Add small random displacement
    np.random.seed(42)  # For reproducibility
    random_displacement = np.random.normal(0, 0.05, translated_positions.shape)
    final_positions = translated_positions + random_displacement

    test_structure = Atoms(
        symbols=shuffled_species,
        positions=final_positions,
        cell=ref_structure.get_cell(),
        pbc=True,
    )

    print(f"Test structure created with {len(test_structure)} atoms")
    print(
        f"Applied translation: [{scaled_translation[0]:.3f}, {scaled_translation[1]:.3f}, {scaled_translation[2]:.3f}] Å"
    )

    # Perform enhanced mapping
    print("\nPerforming enhanced atom mapping...")
    mapping, cost, shift_vector, quality = create_enhanced_atom_mapping(
        ref_structure, test_structure, optimize_shift=True, origin_alignment=True
    )

    print(f"Mapping completed!")
    print(f"Total mapping cost: {cost:.6f} Å")
    print(
        f"Optimized shift vector: [{shift_vector[0]:.6f}, {shift_vector[1]:.6f}, {shift_vector[2]:.6f}] Å"
    )
    print(f"Shift magnitude: {np.linalg.norm(shift_vector):.6f} Å")

    # Display quality metrics
    print(f"\nQuality Metrics:")
    print(f"  Mean distance: {quality['mean_distance']:.6f} Å")
    print(f"  Max distance: {quality['max_distance']:.6f} Å")
    print(f"  Std distance: {quality['std_distance']:.6f} Å")
    print(f"  Atoms > 0.1Å: {quality['atoms_above_01angstrom']}")
    print(f"  Atoms > 0.5Å: {quality['atoms_above_05angstrom']}")

    # Create detailed output
    print("\nGenerating detailed analysis output...")
    analyzer = MappingAnalyzer(
        ref_structure, test_structure, mapping, shift_vector, quality
    )

    # Save to data/mapping directory
    output_file = "data/mapping/enhanced_mapping_example.txt"
    analyzer.save_detailed_output(output_file)

    print(f"Detailed analysis saved to: {output_file}")
    print()


def demonstrate_synthetic_mapping():
    """
    Demonstrate enhanced mapping with synthetic data when real data is unavailable.
    """
    print("Creating synthetic structures for demonstration...")

    # Create a simple perovskite-like structure
    cell = np.eye(3) * 4.0  # 4Å cubic cell

    # Simple 5-atom structure
    positions = np.array(
        [
            [0.0, 0.0, 0.0],  # A-site
            [2.0, 2.0, 2.0],  # B-site
            [1.0, 1.0, 3.0],  # O1
            [3.0, 1.0, 1.0],  # O2
            [1.0, 3.0, 1.0],  # O3
        ]
    )
    species = ["A", "B", "O", "O", "O"]

    ref_structure = Atoms(symbols=species, positions=positions, cell=cell, pbc=True)

    # Create test structure (shuffled + translated + displaced)
    shuffled_indices = np.random.permutation(len(ref_structure))
    shuffled_positions = positions[shuffled_indices]
    shuffled_species = [species[i] for i in shuffled_indices]

    # Apply translation
    translation = np.array([0.5, 0.5, 0.5])
    translated_positions = shuffled_positions + translation

    # Add small displacement
    np.random.seed(42)
    displacement = np.random.normal(0, 0.02, translated_positions.shape)
    final_positions = translated_positions + displacement

    test_structure = Atoms(
        symbols=shuffled_species, positions=final_positions, cell=cell, pbc=True
    )

    print(f"Created synthetic structures with {len(ref_structure)} atoms")

    # Perform enhanced mapping
    mapping, cost, shift_vector, quality = create_enhanced_atom_mapping(
        ref_structure, test_structure, optimize_shift=True, origin_alignment=True
    )

    print(f"Mapping cost: {cost:.6f} Å")
    print(
        f"Shift vector: [{shift_vector[0]:.3f}, {shift_vector[1]:.3f}, {shift_vector[2]:.3f}] Å"
    )
    print(f"Mean distance: {quality['mean_distance']:.6f} Å")

    # Generate output
    analyzer = MappingAnalyzer(
        ref_structure, test_structure, mapping, shift_vector, quality
    )
    output_file = "data/mapping/synthetic_mapping_example.txt"
    analyzer.save_detailed_output(output_file)
    print(f"Analysis saved to: {output_file}")


def demonstrate_backward_compatibility():
    """
    Demonstrate backward compatibility with existing functions.
    """
    print("=" * 60)
    print("Backward Compatibility Demonstration")
    print("=" * 60)

    # Create simple test structures
    positions1 = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    positions2 = np.array([[0.1, 0.0, 0.0], [1.1, 0.0, 0.0]])

    structure1 = Atoms(symbols=["H", "H"], positions=positions1, cell=np.eye(3) * 5.0)
    structure2 = Atoms(symbols=["H", "H"], positions=positions2, cell=np.eye(3) * 5.0)

    # Test original create_atom_mapping function (should work unchanged)
    from phonproj.core.structure_analysis import create_atom_mapping

    print("Testing original create_atom_mapping function...")
    try:
        mapping, cost = create_atom_mapping(structure1, structure2)
        print(f"✓ Original function works: mapping={mapping}, cost={cost:.6f}")
    except Exception as e:
        print(f"✗ Original function failed: {e}")

    # Test enhanced function with default parameters (should behave similarly)
    print("\nTesting enhanced create_enhanced_atom_mapping with defaults...")
    try:
        enhanced_mapping, enhanced_cost, shift_vector, quality = (
            create_enhanced_atom_mapping(
                structure1, structure2, optimize_shift=False, origin_alignment=False
            )
        )
        print(
            f"✓ Enhanced function works: mapping={enhanced_mapping}, cost={enhanced_cost:.6f}"
        )
        print(f"  Shift vector: {shift_vector}")
        print(f"  Quality metrics: {quality}")
    except Exception as e:
        print(f"✗ Enhanced function failed: {e}")

    print()


def main():
    """
    Main function to run all demonstrations.
    """
    print("Enhanced Structure Mapping Example")
    print("=" * 60)
    print("This example demonstrates the enhanced structure mapping functionality")
    print("including PBC distance calculation, origin alignment, and detailed output.")
    print()

    # Run all demonstrations
    #demonstrate_pbc_distance_calculation()
    #demonstrate_origin_alignment()
    demonstrate_enhanced_mapping()
    demonstrate_backward_compatibility()

    print("=" * 60)
    print("Example completed!")
    print("Check the data/mapping/ directory for detailed output files.")
    print("=" * 60)


if __name__ == "__main__":
    main()
