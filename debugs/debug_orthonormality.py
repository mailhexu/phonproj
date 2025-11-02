#!/usr/bin/env python3
"""
Debug script to test orthonormality of phonon modes and understand
the projection coefficient sum issue.
"""

import numpy as np
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from phonproj.modes import PhononModes


def load_phonon_modes():
    """Load phonon modes for testing"""
    print("🔹 Loading phonon modes...")

    data_dir = project_root / "data" / "yajundata" / "0.02-P4mmm-PTO"
    qpoints_16x1x1 = []
    for i in range(16):
        qpoints_16x1x1.append([i / 16.0, 0.0, 0.0])
    qpoints = np.array(qpoints_16x1x1)

    phonon_modes = PhononModes.from_phonopy_directory(str(data_dir), qpoints=qpoints)

    print(
        f"   ✓ Loaded {phonon_modes.n_qpoints} q-points with {phonon_modes.n_modes} modes each"
    )
    print(f"   ✓ Primitive cell: {len(phonon_modes.primitive_cell)} atoms")

    return phonon_modes


def test_individual_qpoint_orthonormality(phonon_modes):
    """Test orthonormality for individual q-points"""
    print("\n🔹 Testing orthonormality for individual q-points...")

    # Test first few q-points
    for q_idx in [0, 1, 2, 15]:  # First, second, third, and last q-point
        try:
            is_orthonormal, max_error, errors = (
                phonon_modes.check_eigenvector_orthonormality(
                    q_index=q_idx, tolerance=1e-10, verbose=False
                )
            )
            qpoint = phonon_modes.qpoints[q_idx]
            print(
                f"   Q-point {q_idx:2d} [{qpoint[0]:5.3f}, {qpoint[1]:5.3f}, {qpoint[2]:5.3f}]:"
            )
            print(f"     Orthonormal: {is_orthonormal}, Max error: {max_error:.2e}")

        except Exception as e:
            print(f"   ❌ Q-point {q_idx} orthonormality check failed: {e}")


def test_eigendisplacement_orthonormality(phonon_modes):
    """Test eigendisplacement orthonormality for individual q-points"""
    print("\n🔹 Testing eigendisplacement orthonormality...")

    # Test first few q-points
    for q_idx in [0, 1, 2, 15]:
        try:
            is_orthonormal, max_error, details = (
                phonon_modes.verify_eigendisplacement_orthonormality(
                    q_index=q_idx, tolerance=1e-8, verbose=False
                )
            )
            qpoint = phonon_modes.qpoints[q_idx]
            print(
                f"   Q-point {q_idx:2d} [{qpoint[0]:5.3f}, {qpoint[1]:5.3f}, {qpoint[2]:5.3f}]:"
            )
            print(
                f"     Eigendisplacements orthonormal: {is_orthonormal}, Max error: {max_error:.2e}"
            )

        except Exception as e:
            print(f"   ❌ Q-point {q_idx} eigendisplacement check failed: {e}")


def test_supercell_orthonormality(phonon_modes):
    """Test orthonormality in supercell context"""
    print("\n🔹 Testing supercell orthonormality...")

    supercell_matrix = np.array([[16, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=int)

    try:
        # Get commensurate q-points
        commensurate_indices = phonon_modes.get_commensurate_qpoints(supercell_matrix)
        print(f"   Found {len(commensurate_indices)} commensurate q-points")

        # Test a few representative q-points
        for i, q_idx in enumerate(commensurate_indices[:3]):
            try:
                is_orthonormal, max_error, errors = (
                    phonon_modes.check_eigenvector_orthonormality(
                        q_index=q_idx, tolerance=1e-10, verbose=False
                    )
                )
                qpoint = phonon_modes.qpoints[q_idx]
                print(
                    f"   Commensurate Q-point {i + 1} (idx {q_idx:2d}) [{qpoint[0]:5.3f}, {qpoint[1]:5.3f}, {qpoint[2]:5.3f}]:"
                )
                print(f"     Orthonormal: {is_orthonormal}, Max error: {max_error:.2e}")

            except Exception as e:
                print(f"   ❌ Commensurate Q-point {q_idx} check failed: {e}")

    except Exception as e:
        print(f"   ❌ Supercell orthonormality test failed: {e}")


def investigate_mass_weighting(phonon_modes):
    """Investigate mass weighting in eigenvectors"""
    print("\n🔹 Investigating mass weighting...")

    # Check atomic masses
    masses = phonon_modes.atomic_masses
    print(f"   Atomic masses: {masses}")
    print(f"   Unique masses: {np.unique(masses)}")

    # Check if eigenvectors are properly mass-weighted
    q_idx = 0
    eigvecs = phonon_modes.eigenvectors[q_idx]  # Shape: (n_modes, n_atoms*3)

    print(f"   Eigenvector shape for q-point 0: {eigvecs.shape}")
    print(f"   Eigenvector dtype: {eigvecs.dtype}")

    # Check normalization of first few eigenvectors
    for mode_idx in range(min(3, eigvecs.shape[0])):
        eigvec = eigvecs[mode_idx]
        norm = np.linalg.norm(eigvec)
        print(f"   Mode {mode_idx}: |eigenvector| = {norm:.6f}")

        # Check if it's normalized with masses
        eigvec_reshaped = eigvec.reshape(-1, 3)  # (n_atoms, 3)
        mass_weighted_norm = 0.0
        for atom_idx in range(len(masses)):
            mass_weighted_norm += masses[atom_idx] * np.sum(
                np.abs(eigvec_reshaped[atom_idx]) ** 2
            )
        mass_weighted_norm = np.sqrt(mass_weighted_norm)
        print(f"   Mode {mode_idx}: mass-weighted norm = {mass_weighted_norm:.6f}")


def check_displacement_projection_logic(phonon_modes):
    """Check the displacement projection logic"""
    print("\n🔹 Checking displacement projection logic...")

    # Create a simple test displacement - just displace first atom in x direction
    supercell_matrix = np.array([[16, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=int)
    n_atoms_supercell = len(phonon_modes.primitive_cell) * int(
        np.round(np.linalg.det(supercell_matrix))
    )

    # Simple test displacement: first atom only
    test_displacement = np.zeros((n_atoms_supercell, 3))
    test_displacement[0, 0] = 1.0  # 1 Å displacement in x

    print(
        f"   Test displacement: {np.count_nonzero(test_displacement)} non-zero entries"
    )
    print(f"   Total displacement magnitude: {np.linalg.norm(test_displacement):.6f} Å")

    try:
        # Project this simple displacement
        projection_table, summary = phonon_modes.decompose_displacement(
            displacement=test_displacement,
            supercell_matrix=supercell_matrix,
            normalize=True,
            tolerance=1e-10,
            print_table=False,
            max_entries=10,
            min_contribution=1e-8,
        )

        total_sum = summary.get("sum_squared_projections", 0.0)
        print(f"   Simple test projection sum: {total_sum:.6f}")

        if total_sum > 0:
            # Show top contributions
            if projection_table:
                coeffs = [
                    (p["q_index"], p["mode_index"], p["squared_coefficient"])
                    for p in projection_table
                ]
                coeffs.sort(key=lambda x: x[2], reverse=True)
                print(f"   Top 3 contributions:")
                for i, (q_idx, mode_idx, coeff_sq) in enumerate(coeffs[:3]):
                    print(
                        f"     {i + 1}. Q={q_idx:2d}, Mode={mode_idx:2d}: {coeff_sq:.6f}"
                    )

    except Exception as e:
        print(f"   ❌ Simple projection test failed: {e}")


def test_projection_normalization_theory(phonon_modes):
    """Test if the issue is with projection normalization theory"""
    print("\n🔹 Testing projection normalization theory...")

    # The theoretical expectation is that if we have an orthonormal complete basis
    # and project any vector onto it, the sum of squared coefficients should equal
    # the squared norm of the original vector.

    # For unit displacement (norm = 1), sum should be 1
    # For our real displacement, sum should equal ||displacement||²

    supercell_matrix = np.array([[16, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=int)

    # Load the real experimental displacement for comparison
    from ase.io import read

    contcar_path = project_root / "data" / "yajundata" / "CONTCAR-a1a2-GS"
    displaced_structure = read(contcar_path)
    if isinstance(displaced_structure, list):
        displaced_structure = displaced_structure[0]

    # Create reference structure
    reference_supercell = phonon_modes.generate_displaced_supercell(
        q_index=0,
        mode_index=0,
        supercell_matrix=supercell_matrix,
        amplitude=0.0,
        return_displacements=False,
    )
    if isinstance(reference_supercell, tuple):
        reference_supercell = reference_supercell[0]

    # Apply alignment (using the working method from earlier)
    from phonproj.core.structure_analysis import find_nearest_atoms

    min_indices, distances, periodic_vectors = find_nearest_atoms(
        reference_supercell, displaced_structure
    )

    from ase import Atoms

    aligned_positions = reference_supercell.positions + periodic_vectors
    aligned_numbers = displaced_structure.numbers[min_indices]

    aligned_displaced = Atoms(
        numbers=aligned_numbers,
        positions=aligned_positions,
        cell=reference_supercell.get_cell(),
        pbc=reference_supercell.get_pbc(),
    )

    displacement = aligned_displaced.positions - reference_supercell.positions

    # Apply minimum image convention
    cell = reference_supercell.cell
    cell_inv = np.linalg.inv(cell.T)
    fractional_disp = np.dot(displacement, cell_inv.T)
    fractional_disp -= np.round(fractional_disp)
    displacement = np.dot(fractional_disp, cell.T)

    displacement_norm_squared = np.sum(displacement**2)
    print(f"   ||displacement||² = {displacement_norm_squared:.6f}")

    # Project displacement
    try:
        projection_table, summary = phonon_modes.decompose_displacement(
            displacement=displacement,
            supercell_matrix=supercell_matrix,
            normalize=True,
            tolerance=1e-10,
            print_table=False,
            max_entries=50,
            min_contribution=1e-8,
        )

        total_sum = summary.get("sum_squared_projections", 0.0)
        print(f"   Σ|coefficients|² = {total_sum:.6f}")
        print(
            f"   Ratio: {total_sum / displacement_norm_squared:.6f} (should be ~1.0 if basis is complete)"
        )

        if abs(total_sum / displacement_norm_squared - 1.0) > 0.1:
            print(
                f"   ❌ Significant deviation from unity ratio suggests incomplete basis or normalization issue"
            )
        else:
            print(f"   ✓ Ratio close to unity suggests basis is approximately complete")

    except Exception as e:
        print(f"   ❌ Projection normalization test failed: {e}")


def main():
    """Main debugging workflow"""
    print("=" * 80)
    print("Debug: Phonon Mode Orthonormality and Projection Analysis")
    print("=" * 80)

    try:
        # Load phonon modes
        phonon_modes = load_phonon_modes()

        # Test orthonormality for individual q-points
        test_individual_qpoint_orthonormality(phonon_modes)

        # Test eigendisplacement orthonormality
        test_eigendisplacement_orthonormality(phonon_modes)

        # Test supercell orthonormality
        test_supercell_orthonormality(phonon_modes)

        # Investigate mass weighting
        investigate_mass_weighting(phonon_modes)

        # Check simple displacement projection
        check_displacement_projection_logic(phonon_modes)

        # Test projection normalization theory
        test_projection_normalization_theory(phonon_modes)

        print("\n" + "=" * 80)
        print("Debug Analysis Complete")
        print("=" * 80)

    except Exception as e:
        print(f"❌ ERROR: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
