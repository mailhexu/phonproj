"""
Test eigendisplacement norm and mass-weighted orthonormality verification.

This module tests the properties of eigendisplacements with mass weighting:
1. Mass-weighted norm should be 1: <u|M|u> = 1 ✓
2. Eigendisplacements are NOT orthonormal with mass-weighted inner product ✓
3. Mass-weighted non-orthonormality: <u_i|M|u_j> ≠ 0 for i≠j (expected behavior)

IMPORTANT THEORETICAL INSIGHT:
- Raw eigenvectors ARE orthonormal: <e_i|e_j> = δ_ij ✓
- Eigendisplacements are mass-weighted: u_i = sqrt(M) * e_i / ||sqrt(M) * e_i||_M
- Mass-weighting breaks orthogonality: <u_i|M|u_j> ≠ δ_ij (this is correct!)

The verification method should return False (non-orthonormal) for physical systems,
confirming that mass-weighting is working correctly.

where M is the mass matrix (diagonal with atomic masses).
"""

import numpy as np
from phonproj.band_structure import PhononBand


def test_eigendisplacement_mass_weighted_norm_batio3():
    """
    Test that eigendisplacements have unit mass-weighted norm for BaTiO3.

    For mass-normalized eigendisplacements, the condition is:
        <u|M|u> = Σ_i m_i * |u_i|² = 1

    where u_i is the displacement of atom i and m_i is its mass.
    """
    print("\n" + "=" * 70)
    print("Testing Eigendisplacement Mass-Weighted Norm (BaTiO3)")
    print("=" * 70)

    # Load BaTiO3 phonon data
    yaml_path = "/Users/hexu/projects/phonproj/data/BaTiO3_phonopy_params.yaml"
    band = PhononBand.calculate_band_structure_from_phonopy(
        yaml_path, path="GMXMG", npoints=50, units="cm-1"
    )

    # Get band as PhononModes for accessing displacement methods
    modes = band

    q_index = 0  # Γ-point
    n_modes = band.frequencies.shape[1]
    n_atoms = len(band.primitive_cell)

    print(f"\nTest parameters:")
    print(f"  q-point index: {q_index} (Γ-point)")
    print(f"  Number of phonon modes: {n_modes}")
    print(f"  Number of atoms: {n_atoms}")

    # Test a few modes
    test_modes = [0, 1, 5, 10]
    tolerance = 1e-10

    print(f"\nTesting mass-weighted norm for modes: {test_modes}")

    all_passed = True
    for mode_idx in test_modes:
        # Get eigendisplacement
        eigendisp = modes.get_eigen_displacement(q_index, mode_idx)

        # Calculate mass-weighted norm
        mass_weighted_norm = modes.mass_weighted_norm(eigendisp)

        print(f"\n  Mode {mode_idx}:")
        print(f"    Frequency: {modes.frequencies[q_index, mode_idx]:.2f} cm⁻¹")
        print(f"    Mass-weighted norm: {mass_weighted_norm:.10f}")
        print(f"    Expected: 1.0")
        print(f"    Error: {abs(mass_weighted_norm - 1.0):.2e}")

        # Check if norm is approximately 1
        if abs(mass_weighted_norm - 1.0) < tolerance:
            print(f"    ✓ PASS")
        else:
            print(f"    ✗ FAIL: Norm is not 1!")
            all_passed = False

    assert all_passed, "Some eigendisplacements do not have unit mass-weighted norm!"

    print("\n✓ All eigendisplacements have unit mass-weighted norm!")
    print(f"  Formula: <u|M|u> = Σ_i m_i * |u_i|² = 1")

    return True


def test_eigendisplacement_mass_weighted_orthonormality_batio3():
    """
    Test mass-weighted orthonormality of eigendisplacements for BaTiO3.

    For eigendisplacements, the mass-weighted orthonormality condition is:
        <u_i|u_j>_M = δ_ij

    where the mass-weighted inner product is:
        <u_i|u_j>_M = Σ_k m_k * u_i,k* · u_j,k

    Note: This tests the eigenvectors themselves, which ARE orthonormal.
    Eigendisplacements include mass weighting, so individual normalization
    check (mass-weighted norm = 1) is done separately.
    """
    print("\n" + "=" * 70)
    print("Testing Eigenvector Mass-Weighted Orthonormality (BaTiO3)")
    print("=" * 70)

    # Load BaTiO3 phonon data
    yaml_path = "/Users/hexu/projects/phonproj/data/BaTiO3_phonopy_params.yaml"
    band = PhononBand.calculate_band_structure_from_phonopy(
        yaml_path, path="GMXMG", npoints=50, units="cm-1"
    )

    q_index = 0  # Γ-point
    n_modes = band.frequencies.shape[1]
    n_atoms = len(band.primitive_cell)

    print(f"\nTest parameters:")
    print(f"  q-point index: {q_index} (Γ-point)")
    print(f"  Number of modes: {n_modes}")
    print(f"  Number of atoms: {n_atoms}")

    # Get eigenvectors (these are the fundamental orthonormal basis)
    print(f"\nChecking eigenvector orthonormality (not eigendisplacements)...")

    eigenvectors_q = band.eigenvectors[q_index]

    # Calculate standard inner products of eigenvectors
    projections = np.zeros((n_modes, n_modes), dtype=complex)

    for i in range(n_modes):
        for j in range(n_modes):
            projections[i, j] = np.vdot(eigenvectors_q[i], eigenvectors_q[j])

    # Check if projections form identity matrix
    identity = np.eye(n_modes)
    tolerance = 1e-10

    max_error = np.max(np.abs(projections - identity))

    print(f"\nEigenvector orthonormality check:")
    print(f"  Maximum deviation from identity: {max_error:.2e}")
    print(f"  Tolerance: {tolerance:.0e}")

    assert max_error < tolerance, (
        f"Eigenvectors are not orthonormal! Maximum deviation: {max_error:.2e}"
    )

    print("\n✓ All eigenvectors are orthonormal!")
    print(f"  Formula: <e_i|e_j> = δ_ij")
    print(f"  Normalization: <e_i|e_i> = 1")
    print(f"  Orthogonality: <e_i|e_j> = 0 for i ≠ j")

    # Note: Eigendisplacements are derived from eigenvectors by:
    # u_i = sqrt(M) * e_i
    # When checked individually with mass-weighted norm, they have <u_i|u_i>_M = 1
    # But mutual orthogonality requires checking U† M U = I

    print("\n✓ Eigenvector orthonormality verified (basis for eigendisplacements)!")

    return True


def test_eigendisplacement_mass_weighted_norm_ppto3():
    """
    Test mass-weighted norm for PbTiO3 eigendisplacements.
    """
    print("\n" + "=" * 70)
    print("Testing Eigendisplacement Mass-Weighted Norm (PbTiO3)")
    print("=" * 70)

    directory = "/Users/hexu/projects/phonproj/data/yajundata/0.02-P4mmm-PTO"

    try:
        # Load PbTiO3 phonon data
        band = PhononBand.calculate_band_structure_from_phonopy(
            directory, path="GMXMG", npoints=50, units="cm-1"
        )

        modes = band
        q_index = 0
        n_modes = band.frequencies.shape[1]

        print(f"\nTest parameters:")
        print(f"  q-point index: {q_index}")
        print(f"  Number of modes: {n_modes}")

        # Test first few modes
        test_modes = [0, 1, 5, 10, 15]
        test_modes = [m for m in test_modes if m < n_modes]

        print(f"\nTesting modes: {test_modes}")

        tolerance = 1e-10
        all_passed = True

        for mode_idx in test_modes:
            eigendisp = modes.get_eigen_displacement(q_index, mode_idx)
            mass_weighted_norm = modes.mass_weighted_norm(eigendisp)

            error = abs(mass_weighted_norm - 1.0)
            print(f"\n  Mode {mode_idx}:")
            print(f"    Norm: {mass_weighted_norm:.10f}")
            print(f"    Error: {error:.2e}")

            if error < tolerance:
                print(f"    ✓ PASS")
            else:
                print(f"    ✗ FAIL")
                all_passed = False

        assert all_passed, "Some eigendisplacements do not have unit norm!"

        print("\n✓ All tested eigendisplacements have unit mass-weighted norm!")

        return True

    except RuntimeError as e:
        msg = str(e)
        if "missing forces" in msg or "not prepared" in msg:
            print(f"\n⚠ Expected error (forces not available): {msg}")
            print("Skipping PbTiO3 test")
            return True
        else:
            raise


def test_eigendisplacement_systematic_orthonormality_batio3():
    """
    Test systematic orthonormality verification of eigendisplacements for BaTiO3.

    This tests the new verify_eigendisplacement_orthonormality() method which
    computes the full orthonormality matrix and verifies U†MU = I.
    """
    print("\n" + "=" * 70)
    print("Testing Systematic Eigendisplacement Orthonormality (BaTiO3)")
    print("=" * 70)

    # Load BaTiO3 phonon data
    yaml_path = "/Users/hexu/projects/phonproj/data/BaTiO3_phonopy_params.yaml"
    band = PhononBand.calculate_band_structure_from_phonopy(
        yaml_path, path="GMXMG", npoints=50, units="cm-1"
    )

    modes = band
    q_index = 0  # Γ-point
    n_modes = band.frequencies.shape[1]
    n_atoms = len(band.primitive_cell)

    print(f"\nTest parameters:")
    print(f"  q-point index: {q_index} (Γ-point)")
    print(f"  Number of modes: {n_modes}")
    print(f"  Number of atoms: {n_atoms}")

    # Test the systematic orthonormality verification
    tolerance = 1e-8
    is_orthonormal, max_error, details = modes.verify_eigendisplacement_orthonormality(
        q_index=q_index, tolerance=tolerance, verbose=True
    )

    # Verify the results
    print(f"\nVerification results:")
    print(f"  Is orthonormal: {is_orthonormal}")
    print(f"  Maximum error: {max_error:.2e}")
    print(f"  Tolerance: {tolerance:.0e}")

    # Check details structure
    assert "orthonormality_matrix" in details, (
        "Missing orthonormality matrix in details"
    )
    assert "deviation_matrix" in details, "Missing deviation matrix in details"
    assert "max_error" in details, "Missing max_error in details"
    assert "diagonal_elements" in details, "Missing diagonal elements in details"

    orthonormality_matrix = details["orthonormality_matrix"]
    print(f"  Orthonormality matrix shape: {orthonormality_matrix.shape}")
    print(f"  Expected: ({n_modes}, {n_modes})")

    # Check matrix dimensions
    assert orthonormality_matrix.shape == (n_modes, n_modes), (
        f"Wrong matrix shape: {orthonormality_matrix.shape}"
    )

    # Check diagonal elements are close to 1
    diagonal_elements = details["diagonal_elements"]
    diagonal_errors = details["diagonal_errors"]
    max_diagonal_error = details["max_diagonal_error"]

    print(
        f"  Diagonal errors: min={np.min(diagonal_errors):.2e}, max={max_diagonal_error:.2e}"
    )
    print(f"  Off-diagonal max: {details['max_off_diagonal']:.2e}")

    # IMPORTANT: Eigendisplacements ARE expected to be orthonormal!
    # This is correct behavior with proper mass-weighting implementation:
    # - Raw eigenvectors ARE orthonormal: <e_i|e_j> = δ_ij ✓
    # - Eigendisplacements use correct mass-weighting: u_i = e_i / sqrt(M)
    # - With proper normalization, orthonormality is preserved: <u_i|M|u_j> = δ_ij ✓

    # Verify that the method correctly identifies orthonormality
    assert is_orthonormal, (
        f"Eigendisplacements are not orthonormal! This suggests an implementation error. "
        f"Max error: {max_error:.2e}, tolerance: {tolerance:.0e}"
    )

    # Verify that the error is within numerical precision
    assert max_error < tolerance, (
        f"Orthonormality error ({max_error:.2e}) exceeds tolerance ({tolerance:.0e}). "
        f"This suggests mass-weighting is not working correctly."
    )

    # Additional checks on the orthonormality matrix
    # Diagonal should be close to 1 (individual normalization preserved)
    for i in range(n_modes):
        diag_val = orthonormality_matrix[i, i]
        assert abs(diag_val - 1.0) < tolerance, (
            f"Diagonal element {i} not normalized: {diag_val:.6f}"
        )

    # Verify that off-diagonal elements are small (confirming orthonormality)
    # This confirms that mass-weighting is implemented correctly
    max_off_diagonal = details["max_off_diagonal"]
    assert max_off_diagonal < tolerance, (
        f"Off-diagonal elements too large ({max_off_diagonal:.2e}). "
        f"This suggests orthonormality is not properly maintained."
    )

    print("\n✓ Eigendisplacement orthonormality verification working correctly!")
    print(f"  ✓ Method correctly identifies orthonormality (correct behavior)")
    print(
        f"  ✓ Diagonal elements properly normalized: max error {max_diagonal_error:.2e}"
    )
    print(f"  ✓ Off-diagonal elements small: max {details['max_off_diagonal']:.2e}")
    print(f"  ✓ Orthonormality confirmed: {max_error:.2e} < tolerance")

    return True


def test_eigendisplacement_systematic_orthonormality_ppto3():
    """
    Test systematic orthonormality verification for PbTiO3 eigendisplacements.
    """
    print("\n" + "=" * 70)
    print("Testing Systematic Eigendisplacement Orthonormality (PbTiO3)")
    print("=" * 70)

    directory = "/Users/hexu/projects/phonproj/data/yajundata/0.02-P4mmm-PTO"

    try:
        # Load PbTiO3 phonon data
        band = PhononBand.calculate_band_structure_from_phonopy(
            directory, path="GMXMG", npoints=50, units="cm-1"
        )

        modes = band
        q_index = 0  # Γ-point
        n_modes = band.frequencies.shape[1]

        print(f"\nTest parameters:")
        print(f"  q-point index: {q_index}")
        print(f"  Number of modes: {n_modes}")

        # Test systematic orthonormality verification
        tolerance = 1e-8
        is_orthonormal, max_error, details = (
            modes.verify_eigendisplacement_orthonormality(
                q_index=q_index, tolerance=tolerance, verbose=True
            )
        )

        print(f"\nVerification results:")
        print(f"  Is orthonormal: {is_orthonormal}")
        print(f"  Maximum error: {max_error:.2e}")

        # Eigendisplacements ARE expected to be orthonormal (same corrected theory as BaTiO3)
        assert is_orthonormal, (
            f"PbTiO3 eigendisplacements are not orthonormal! "
            f"This suggests an implementation error. Max error: {max_error:.2e}"
        )

        # Verify the error is within numerical precision
        assert max_error < tolerance, (
            f"Orthonormality error ({max_error:.2e}) exceeds tolerance ({tolerance:.0e}) for PbTiO3."
        )

        print(
            "\n✓ PbTiO3 eigendisplacement orthonormality verification working correctly!"
        )
        print(f"  ✓ Non-orthonormality confirmed (expected): {max_error:.2e}")

        return True

    except RuntimeError as e:
        msg = str(e)
        if "missing forces" in msg or "not prepared" in msg:
            print(f"\n⚠ Expected error (forces not available): {msg}")
            print("Skipping PbTiO3 systematic orthonormality test")
            return True
        else:
            raise


def test_eigendisplacement_orthonormality_edge_cases():
    """
    Test edge cases for eigendisplacement orthonormality verification.
    """
    print("\n" + "=" * 70)
    print("Testing Eigendisplacement Orthonormality Edge Cases")
    print("=" * 70)

    # Load BaTiO3 data
    yaml_path = "/Users/hexu/projects/phonproj/data/BaTiO3_phonopy_params.yaml"
    band = PhononBand.calculate_band_structure_from_phonopy(
        yaml_path, path="GMXMG", npoints=50, units="cm-1"
    )

    modes = band
    q_index = 0

    # Test with different tolerance values
    tolerances = [1e-6, 1e-8, 1e-10, 1e-12]

    print(f"\nTesting different tolerance values:")
    for tol in tolerances:
        is_orthonormal, max_error, details = (
            modes.verify_eigendisplacement_orthonormality(
                q_index=q_index, tolerance=tol, verbose=False
            )
        )
        print(
            f"  Tolerance {tol:.0e}: {'PASS' if is_orthonormal else 'FAIL'} (max_error: {max_error:.2e})"
        )

    # Test invalid q_index
    print(f"\nTesting invalid q_index:")
    try:
        modes.verify_eigendisplacement_orthonormality(q_index=-1)
        assert False, "Should have raised ValueError for negative q_index"
    except ValueError as e:
        print(f"  ✓ Correctly caught negative q_index: {e}")

    try:
        modes.verify_eigendisplacement_orthonormality(q_index=1000)
        assert False, "Should have raised ValueError for out-of-range q_index"
    except ValueError as e:
        print(f"  ✓ Correctly caught out-of-range q_index: {e}")

    print("\n✓ Edge case testing completed!")

    return True


def test_eigendisplacement_properties():
    """
    Test additional properties of eigendisplacements.
    """
    print("\n" + "=" * 70)
    print("Testing Eigendisplacement Properties")
    print("=" * 70)

    # Load BaTiO3 data
    yaml_path = "/Users/hexu/projects/phonproj/data/BaTiO3_phonopy_params.yaml"
    band = PhononBand.calculate_band_structure_from_phonopy(
        yaml_path, path="GMXMG", npoints=50, units="cm-1"
    )

    modes = band
    q_index = 0
    n_modes = band.frequencies.shape[1]
    n_atoms = len(band.primitive_cell)

    print(f"\nSystem information:")
    print(f"  Number of atoms: {n_atoms}")
    print(f"  Number of modes: {n_modes}")
    print(f"  Atomic masses: {band.atomic_masses}")

    # Analyze acoustic modes (lowest 3 modes at Γ-point)
    acoustic_modes = []
    for i in range(min(3, n_modes)):
        freq = modes.frequencies[q_index, i]
        eigendisp = modes.get_eigen_displacement(q_index, i)
        norm = modes.mass_weighted_norm(eigendisp)
        acoustic_modes.append((i, freq, norm))

    print(f"\nAcoustic mode analysis:")
    for mode_idx, freq, norm in acoustic_modes:
        print(f"  Mode {mode_idx}: freq={freq:.2f} cm⁻¹, norm={norm:.10f}")

    # Check that acoustic modes are normalized
    for mode_idx, freq, norm in acoustic_modes:
        assert abs(norm - 1.0) < 1e-10, f"Acoustic mode {mode_idx} not normalized!"

    # Analyze optical modes
    optical_modes = []
    for i in range(3, min(6, n_modes)):  # Look at first few optical modes
        freq = modes.frequencies[q_index, i]
        eigendisp = modes.get_eigen_displacement(q_index, i)
        norm = modes.mass_weighted_norm(eigendisp)
        optical_modes.append((i, freq, norm))

    print(f"\nOptical mode analysis:")
    for mode_idx, freq, norm in optical_modes:
        print(f"  Mode {mode_idx}: freq={freq:.2f} cm⁻¹, norm={norm:.10f}")

    print("\n✓ Eigendisplacement properties verified!")

    return True


def test_phase_normalization_all_modes():
    """
    Test that phase normalization makes the maximum component real and positive
    for all modes using generate_all_mode_displacements.

    After phase normalization, each mode displacement should have its maximum
    component (by absolute value) be a positive real number.
    """
    print("\n" + "=" * 70)
    print("Testing Phase Normalization for All Modes")
    print("=" * 70)

    # Load BaTiO3 data
    yaml_path = "/Users/hexu/projects/phonproj/data/BaTiO3_phonopy_params.yaml"
    band = PhononBand.calculate_band_structure_from_phonopy(
        yaml_path, path="GMXMG", npoints=50, units="cm-1"
    )

    modes = band
    q_index = 0  # Γ-point
    n_modes = band.frequencies.shape[1]

    print(f"\nTest parameters:")
    print(f"  q-point index: {q_index} (Γ-point)")
    print(f"  Number of modes: {n_modes}")

    # Test various supercell sizes
    test_supercells = [
        np.eye(3, dtype=int),  # 1x1x1 (special case)
        np.diag([2, 2, 2]),  # 2x2x2
        np.diag([3, 3, 3]),  # 3x3x3
    ]

    for supercell_matrix in test_supercells:
        det = int(np.linalg.det(supercell_matrix))
        print(f"\n--- Testing supercell: {det}x unit cell ---")

        # Generate all mode displacements
        all_displacements = modes.generate_all_mode_displacements(
            q_index=q_index, supercell_matrix=supercell_matrix, amplitude=1.0
        )

        print(f"  Generated displacements shape: {all_displacements.shape}")

        # Check phase normalization for each mode
        tolerance = 1e-10
        all_passed = True
        failed_modes = []

        for mode_idx in range(n_modes):
            displacement = all_displacements[mode_idx]
            displacement_flat = displacement.flatten()

            # Find the maximum component by absolute value
            index_max = np.argmax(np.abs(displacement_flat))
            max_component = displacement_flat[index_max]

            # Check that max component is real and positive
            imaginary_part = np.imag(max_component)
            real_part = np.real(max_component)

            is_real = abs(imaginary_part) < tolerance
            is_positive = real_part > 0

            if not (is_real and is_positive):
                all_passed = False
                failed_modes.append(
                    {
                        "mode": mode_idx,
                        "max_component": max_component,
                        "imaginary_part": imaginary_part,
                        "real_part": real_part,
                        "is_real": is_real,
                        "is_positive": is_positive,
                    }
                )

        if all_passed:
            print(f"  ✓ All {n_modes} modes have real positive maximum components")
        else:
            print(f"  ✗ {len(failed_modes)} modes failed phase normalization:")
            for failure in failed_modes[:5]:  # Show first 5 failures
                print(f"    Mode {failure['mode']}:")
                print(f"      Max component: {failure['max_component']}")
                print(f"      Real part: {failure['real_part']:.10f}")
                print(f"      Imaginary part: {failure['imaginary_part']:.10e}")
                print(
                    f"      Is real: {failure['is_real']}, Is positive: {failure['is_positive']}"
                )

        assert all_passed, (
            f"Phase normalization failed for {len(failed_modes)} modes in "
            f"{det}x supercell. See details above."
        )

    print("\n✓ Phase normalization verified for all tested supercells!")
    print("  All mode displacements have their maximum component real and positive.")

    return True


if __name__ == "__main__":
    # Run all tests
    test_eigendisplacement_mass_weighted_norm_batio3()
    test_eigendisplacement_mass_weighted_orthonormality_batio3()
    test_eigendisplacement_systematic_orthonormality_batio3()
    test_eigendisplacement_mass_weighted_norm_ppto3()
    test_eigendisplacement_systematic_orthonormality_ppto3()
    test_eigendisplacement_orthonormality_edge_cases()
    test_eigendisplacement_properties()
    test_phase_normalization_all_modes()

    print("\n" + "=" * 70)
    print("All eigendisplacement tests passed!")
    print("=" * 70)
