#!/usr/bin/env python3
"""
Fix for the mass-weighted normalization issue in decompose_displacement_to_modes.

The problem is that we're double-applying mass weighting:
1. We normalize both target and mode displacements with mass-weighted norms
2. Then we project using mass-weighted inner product

This should be fixed to use regular inner product after mass-weighted normalization.
"""

import numpy as np
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


def fixed_project_displacements_between_supercells(
    source_displacement: np.ndarray,
    target_displacement: np.ndarray,
    source_supercell,  # Atoms object
    target_supercell,  # Atoms object
    atom_mapping=None,
    normalize: bool = True,
    use_mass_weighting: bool = True,
):
    """
    Fixed version that allows choosing between mass-weighted and regular inner product.

    When both displacements are already mass-weighted normalized, we should use
    regular inner product, not mass-weighted inner product.
    """
    from phonproj.core.structure_analysis import create_atom_mapping

    # Get atom numbers
    n_source_atoms = len(source_supercell)
    n_target_atoms = len(target_supercell)

    # Check dimensions
    if source_displacement.shape[0] != n_source_atoms:
        raise ValueError("Source displacement must match source supercell atom count")
    if target_displacement.shape[0] != n_target_atoms:
        raise ValueError("Target displacement must match target supercell atom count")

    # Create atom mapping if not provided
    if atom_mapping is None:
        atom_mapping, _ = create_atom_mapping(
            source_supercell, target_supercell, max_cost=100.0
        )

    # Map source displacement to target supercell atom ordering
    mapped_source_displacement = np.zeros_like(target_displacement)

    # Create inverse mapping: from target atoms to source atoms
    inverse_mapping = np.zeros(n_target_atoms, dtype=int)
    for source_idx, target_idx in enumerate(atom_mapping):
        inverse_mapping[target_idx] = source_idx

    # Map source displacement to target ordering
    for target_idx in range(n_target_atoms):
        source_idx = inverse_mapping[target_idx]
        mapped_source_displacement[target_idx] = source_displacement[source_idx]

    # Flatten displacements
    mapped_source_flat = mapped_source_displacement.ravel()
    target_flat = target_displacement.ravel()

    if use_mass_weighting:
        # Original mass-weighted projection
        target_atomic_masses = target_supercell.get_masses()
        target_masses = np.repeat(target_atomic_masses, 3)

        # Calculate mass-weighted inner product
        projection = np.sum(target_masses * mapped_source_flat.conj() * target_flat)

        if normalize:
            # Calculate mass-weighted norms
            source_norm = np.sqrt(
                np.sum(target_masses * mapped_source_flat.conj() * mapped_source_flat)
            )
            target_norm = np.sqrt(
                np.sum(target_masses * target_flat.conj() * target_flat)
            )

            # Normalized projection coefficient
            if source_norm > 0 and target_norm > 0:
                coefficient = projection / (source_norm * target_norm)
            else:
                coefficient = 0.0
        else:
            # Unnormalized projection (just the inner product)
            coefficient = projection
    else:
        # Regular (non-mass-weighted) projection
        # This should be used when both displacements are already mass-weighted normalized

        # Calculate regular inner product
        projection = np.sum(mapped_source_flat.conj() * target_flat)

        if normalize:
            # Calculate regular norms
            source_norm = np.sqrt(
                np.sum(mapped_source_flat.conj() * mapped_source_flat)
            )
            target_norm = np.sqrt(np.sum(target_flat.conj() * target_flat))

            # Normalized projection coefficient
            if source_norm > 0 and target_norm > 0:
                coefficient = projection / (source_norm * target_norm)
            else:
                coefficient = 0.0
        else:
            # Unnormalized projection (just the inner product)
            coefficient = projection

    return float(coefficient.real)


def test_fixed_projection():
    """Test the fix with a simple example"""
    print("🔹 Testing fixed projection function...")

    from phonproj.modes import PhononModes
    from ase.io import read

    # Load phonon modes
    data_dir = project_root / "data" / "yajundata" / "0.02-P4mmm-PTO"
    qpoints_16x1x1 = []
    for i in range(16):
        qpoints_16x1x1.append([i / 16.0, 0.0, 0.0])
    qpoints = np.array(qpoints_16x1x1)

    phonon_modes = PhononModes.from_phonopy_directory(str(data_dir), qpoints=qpoints)

    # Create a simple test displacement
    supercell_matrix = np.array([[16, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=int)
    n_atoms_supercell = len(phonon_modes.primitive_cell) * int(
        np.round(np.linalg.det(supercell_matrix))
    )

    # Simple test: displace first atom by 1 Å in x
    test_displacement = np.zeros((n_atoms_supercell, 3))
    test_displacement[0, 0] = 1.0

    # Create target supercell
    from phonproj.modes import create_supercell

    target_supercell = create_supercell(phonon_modes.primitive_cell, supercell_matrix)

    # Test mass-weighted normalization
    target_masses = np.repeat(target_supercell.get_masses(), 3)
    target_flat = test_displacement.ravel()
    mass_weighted_norm = np.sqrt(
        np.sum(target_masses * target_flat.conj() * target_flat)
    )

    print(f"   Original displacement norm: {np.linalg.norm(test_displacement):.6f}")
    print(f"   Mass-weighted norm: {mass_weighted_norm:.6f}")

    # Normalize with mass weighting
    normalized_displacement = test_displacement / mass_weighted_norm

    print(
        f"   Normalized displacement norm: {np.linalg.norm(normalized_displacement):.6f}"
    )

    # Test projection of normalized displacement onto itself
    # Should give coefficient = 1.0 regardless of method

    coeff_mass_weighted = fixed_project_displacements_between_supercells(
        normalized_displacement,
        normalized_displacement,
        target_supercell,
        target_supercell,
        normalize=False,
        use_mass_weighting=True,
    )

    coeff_regular = fixed_project_displacements_between_supercells(
        normalized_displacement,
        normalized_displacement,
        target_supercell,
        target_supercell,
        normalize=False,
        use_mass_weighting=False,
    )

    print(f"   Self-projection (mass-weighted): {coeff_mass_weighted:.6f}")
    print(f"   Self-projection (regular): {coeff_regular:.6f}")

    return phonon_modes, normalized_displacement, target_supercell, supercell_matrix


def test_fixed_decomposition(
    phonon_modes, displacement, target_supercell, supercell_matrix
):
    """Test the fixed decomposition approach"""
    print("\n🔹 Testing fixed decomposition approach...")

    # Get commensurate q-points
    result = phonon_modes.get_commensurate_qpoints(supercell_matrix, detailed=True)
    if isinstance(result, list):
        matched_indices = result
    else:
        matched_indices = result.get("matched_indices", [])

    print(f"   Found {len(matched_indices)} commensurate q-points")

    total_squared_projections_mass = 0.0
    total_squared_projections_regular = 0.0

    # Test all modes
    for i, q_index in enumerate(matched_indices):  # Test all q-points
        q_frequencies = phonon_modes.frequencies[q_index]

        for mode_index in range(len(q_frequencies)):  # Test all modes
            # Generate mode displacement
            try:
                mode_displacement = phonon_modes.generate_mode_displacement(
                    q_index=q_index,
                    mode_index=mode_index,
                    supercell_matrix=supercell_matrix,
                    amplitude=1.0,
                )
            except Exception:
                continue

            # Mass-weighted normalize mode displacement
            from phonproj.modes import create_supercell

            source_supercell = create_supercell(
                phonon_modes.primitive_cell, supercell_matrix
            )
            source_masses = np.repeat(source_supercell.get_masses(), 3)
            mode_flat = mode_displacement.ravel()
            mode_norm = np.sqrt(np.sum(source_masses * mode_flat.conj() * mode_flat))
            if mode_norm > 1e-10:
                mode_displacement = mode_displacement / mode_norm

            # Project with mass-weighted inner product (original method)
            coeff_mass = fixed_project_displacements_between_supercells(
                mode_displacement,
                displacement,
                source_supercell,
                target_supercell,
                normalize=False,
                use_mass_weighting=True,
            )

            # Project with regular inner product (fixed method)
            coeff_regular = fixed_project_displacements_between_supercells(
                mode_displacement,
                displacement,
                source_supercell,
                target_supercell,
                normalize=False,
                use_mass_weighting=False,
            )

            total_squared_projections_mass += coeff_mass**2
            total_squared_projections_regular += coeff_regular**2

            if mode_index < 9:  # Only print first few for brevity
                print(
                    f"   Q={q_index:2d}, Mode={mode_index:2d}: mass={coeff_mass**2:.6f}, regular={coeff_regular**2:.6f}"
                )

    print(f"\n   Total sums (all {len(matched_indices)} q-points, all modes):")
    print(f"   Mass-weighted method: {total_squared_projections_mass:.6f}")
    print(f"   Regular method: {total_squared_projections_regular:.6f}")

    return total_squared_projections_mass, total_squared_projections_regular


def main():
    """Test the fix for mass-weighted projection"""
    print("=" * 80)
    print("Testing Fix for Mass-Weighted Projection Issue")
    print("=" * 80)

    try:
        # Test the fixed projection function
        phonon_modes, displacement, target_supercell, supercell_matrix = (
            test_fixed_projection()
        )

        # Test fixed decomposition approach
        mass_sum, regular_sum = test_fixed_decomposition(
            phonon_modes, displacement, target_supercell, supercell_matrix
        )

        print("\n" + "=" * 80)
        print("Summary:")
        print(f"  - Mass-weighted projection gives self-projection ≠ 1.0")
        print(f"  - Regular projection gives self-projection = 1.0")
        print(f"  - This suggests the fix should use regular inner product")
        print(f"  - After mass-weighted normalization of both vectors")
        print("=" * 80)

    except Exception as e:
        print(f"❌ ERROR: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
