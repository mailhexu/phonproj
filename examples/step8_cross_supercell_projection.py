#!/usr/bin/env python3
"""
Step 8 Cross-Supercell Displacement Projection Example

This example demonstrates how to project displacement patterns between different
supercells that may have different atom ordering and positions due to periodic
boundary conditions. This is particularly useful for:

1. Comparing phonon modes between different supercell sizes
2. Analyzing mode localization across structural variants
3. Studying the effect of translation and atom reordering on mode similarity

Key Features Demonstrated:
- Displacement-to-displacement projection between different supercells
- Handling of periodic boundary condition effects
- Robust atom mapping for reordered structures
- Mass-weighted projection calculations
- Both normalized and unnormalized projection coefficients
"""

import numpy as np
import copy
from pathlib import Path

from phonproj.modes import PhononModes
from phonproj.core.structure_analysis import project_displacements_between_supercells


def main():
    """
    Demonstrate Step 8 cross-supercell displacement projection functionality.
    """
    print("=" * 60)
    print("Step 8: Cross-Supercell Displacement Projection Example")
    print("=" * 60)

    # Load BaTiO3 phonon data
    data_path = Path(__file__).parent.parent / "data" / "BaTiO3_phonopy_params.yaml"
    if not data_path.exists():
        print(f"Error: Data file not found at {data_path}")
        print(
            "Please ensure BaTiO3_phonopy_params.yaml is available in the data directory."
        )
        return

    print(f"\n1. Loading BaTiO3 phonon data from: {data_path}")
    gamma_qpoint = np.array([[0.0, 0.0, 0.0]])
    modes = PhononModes.from_phonopy_yaml(str(data_path), gamma_qpoint)
    print(f"   - Loaded {modes.n_modes} modes at {modes.n_qpoints} q-point(s)")

    # Use 2x2x2 supercell for demonstrations
    supercell_matrix = np.eye(3, dtype=int) * 2
    print(
        f"\n2. Using supercell matrix: {supercell_matrix[0, 0]}x{supercell_matrix[1, 1]}x{supercell_matrix[2, 2]}"
    )

    # Generate a reference displacement (mode 6 - first non-acoustic mode)
    mode_index = 6
    print(f"\n3. Generating reference displacement for mode {mode_index}")
    reference_displacement = modes.generate_mode_displacement(
        q_index=0,
        mode_index=mode_index,
        supercell_matrix=supercell_matrix,
        amplitude=1.0,
    )

    # Create reference supercell
    reference_supercell_result = modes.generate_displaced_supercell(
        q_index=0,
        mode_index=mode_index,
        supercell_matrix=supercell_matrix,
        amplitude=0.0,
        return_displacements=False,
    )
    # Extract Atoms object from result (handle both tuple and direct return)
    if isinstance(reference_supercell_result, tuple):
        reference_supercell = reference_supercell_result[0]
    else:
        reference_supercell = reference_supercell_result

    print(f"   - Reference supercell has {len(reference_supercell)} atoms")
    print(f"   - Displacement shape: {reference_displacement.shape}")

    # Demonstrate various projection scenarios
    print("\n" + "=" * 50)
    print("PROJECTION SCENARIOS")
    print("=" * 50)

    # Scenario 1: Identical supercells and displacements
    print("\n🔹 Scenario 1: Identical supercells and displacements")
    target_supercell_1 = copy.deepcopy(reference_supercell)

    # Normalized projection (should be 1.0)
    coeff_normalized = project_displacements_between_supercells(
        source_displacement=reference_displacement,
        target_displacement=reference_displacement,
        source_supercell=reference_supercell,
        target_supercell=target_supercell_1,
        normalize=True,
    )

    # Unnormalized projection (mass-weighted norm squared)
    coeff_unnormalized = project_displacements_between_supercells(
        source_displacement=reference_displacement,
        target_displacement=reference_displacement,
        source_supercell=reference_supercell,
        target_supercell=target_supercell_1,
        normalize=False,
    )

    print(f"   - Normalized coefficient:   {coeff_normalized:.8f} (should be ~1.0)")
    print(
        f"   - Unnormalized coefficient: {coeff_unnormalized:.8f} (mass-weighted norm²)"
    )

    # Scenario 2: Translated supercell
    print("\n🔹 Scenario 2: Translated supercell")
    target_supercell_2 = copy.deepcopy(reference_supercell)
    translation_vector = np.array([0.5, 0.3, 0.2])
    target_supercell_2.positions = target_supercell_2.positions + translation_vector
    target_supercell_2.wrap()  # Apply periodic boundary conditions

    coeff_translated = project_displacements_between_supercells(
        source_displacement=reference_displacement,
        target_displacement=reference_displacement,  # Same displacement pattern
        source_supercell=reference_supercell,
        target_supercell=target_supercell_2,
        normalize=True,
    )

    print(f"   - Translation: {translation_vector}")
    print(f"   - Projection coefficient: {coeff_translated:.8f} (should be ~1.0)")

    # Scenario 3: Shuffled atoms with corresponding displacement reordering
    print("\n🔹 Scenario 3: Shuffled atoms and displacements")
    np.random.seed(42)  # For reproducibility
    n_atoms = len(reference_supercell)
    shuffle_indices = np.arange(n_atoms)
    np.random.shuffle(shuffle_indices)

    # Create shuffled supercell
    target_supercell_3 = copy.deepcopy(reference_supercell)
    target_supercell_3.positions = target_supercell_3.positions[shuffle_indices]
    target_supercell_3.numbers = target_supercell_3.numbers[shuffle_indices]

    # Create correspondingly shuffled displacement
    shuffled_displacement = reference_displacement[shuffle_indices]

    coeff_shuffled = project_displacements_between_supercells(
        source_displacement=reference_displacement,
        target_displacement=shuffled_displacement,
        source_supercell=reference_supercell,
        target_supercell=target_supercell_3,
        normalize=True,
    )

    print(f"   - Shuffled {n_atoms} atoms randomly")
    print(f"   - Projection coefficient: {coeff_shuffled:.8f} (should be ~1.0)")

    # Scenario 4: Combined translation and shuffling
    print("\n🔹 Scenario 4: Combined translation and shuffling")
    target_supercell_4 = copy.deepcopy(reference_supercell)

    # Apply shuffling
    target_supercell_4.positions = target_supercell_4.positions[shuffle_indices]
    target_supercell_4.numbers = target_supercell_4.numbers[shuffle_indices]

    # Apply translation
    translation_vector_2 = np.array([0.4, 0.6, 0.1])
    target_supercell_4.positions = target_supercell_4.positions + translation_vector_2
    target_supercell_4.wrap()

    coeff_combined = project_displacements_between_supercells(
        source_displacement=reference_displacement,
        target_displacement=shuffled_displacement,
        source_supercell=reference_supercell,
        target_supercell=target_supercell_4,
        normalize=True,
    )

    print(f"   - Translation: {translation_vector_2}")
    print(f"   - Also shuffled atoms")
    print(f"   - Projection coefficient: {coeff_combined:.8f} (should be ~1.0)")

    # Scenario 5: Orthogonal displacements
    print("\n🔹 Scenario 5: Orthogonal displacements (different modes)")
    mode_index_2 = 7  # Different mode
    orthogonal_displacement = modes.generate_mode_displacement(
        q_index=0,
        mode_index=mode_index_2,
        supercell_matrix=supercell_matrix,
        amplitude=1.0,
    )

    coeff_orthogonal = project_displacements_between_supercells(
        source_displacement=reference_displacement,
        target_displacement=orthogonal_displacement,
        source_supercell=reference_supercell,
        target_supercell=reference_supercell,  # Same supercell
        normalize=True,
    )

    print(f"   - Mode {mode_index} vs Mode {mode_index_2}")
    print(f"   - Projection coefficient: {coeff_orthogonal:.8f} (should be ~0.0)")

    # Demonstrate convenience method
    print("\n🔹 Convenience Method: PhononModes.project_displacement_to_supercell()")
    coeff_convenience = modes.project_displacement_to_supercell(
        source_displacement=reference_displacement,
        source_supercell=reference_supercell,
        target_supercell=target_supercell_1,
        normalize=True,
    )

    print(f"   - Using convenience method: {coeff_convenience:.8f}")
    print(
        f"   - Matches direct function call: {abs(coeff_convenience - coeff_normalized) < 1e-12}"
    )

    # Summary
    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    print("\nStep 8 projection functionality successfully demonstrated:")
    print(f"✓ Identical structures:     {coeff_normalized:.6f}")
    print(f"✓ Translated structure:     {coeff_translated:.6f}")
    print(f"✓ Shuffled atoms:           {coeff_shuffled:.6f}")
    print(f"✓ Combined transformations: {coeff_combined:.6f}")
    print(f"✓ Orthogonal modes:         {coeff_orthogonal:.6f}")
    print(f"✓ Convenience method:       {coeff_convenience:.6f}")

    print("\nKey capabilities:")
    print("• Robust atom mapping between different supercell arrangements")
    print("• Mass-weighted projection calculations")
    print("• Handling of periodic boundary condition effects")
    print("• Support for both normalized and unnormalized projections")
    print("• Integration with PhononModes workflow")

    print(f"\nExample completed successfully! 🎉")


if __name__ == "__main__":
    main()
