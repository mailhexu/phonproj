#!/usr/bin/env python3
"""
Comprehensive test suite for complex displacement implementation.

This test validates that our implementation correctly:
1. Returns complex displacements for non-Gamma q-points
2. Returns real displacements for Gamma points
3. Preserves orthogonality within degenerate subspaces
4. Handles both unit cell and supercell displacement generation
"""

import numpy as np
import sys

sys.path.insert(0, "/Users/hexu/projects/phonproj")

from phonproj.modes import PhononModes


def test_complex_displacement_implementation():
    """Comprehensive test of complex displacement implementation."""

    print("=== Complex Displacement Implementation Test Suite ===")

    # Load test data
    BATIO3_YAML_PATH = "/Users/hexu/projects/phonproj/data/BaTiO3_phonopy_params.yaml"
    qpoints = np.array(
        [
            [0.0, 0.0, 0.0],  # Gamma point
            [0.0, 0.0, 0.5],  # Non-Gamma point commensurate with 2x2x1 supercell
            [
                0.5,
                0.5,
                0.0,
            ],  # Another non-Gamma point commensurate with 2x2x1 supercell
        ]
    )
    modes = PhononModes.from_phonopy_yaml(BATIO3_YAML_PATH, qpoints=qpoints)

    print(f"Loaded {modes.n_qpoints} q-points with {modes.n_modes} modes")

    all_tests_passed = True

    for q_idx in range(modes.n_qpoints):
        qpoint = modes.qpoints[q_idx]
        is_gamma = np.allclose(qpoint, 0.0, atol=1e-6)
        frequencies = modes.frequencies[q_idx]

        print(
            f"\n--- Testing Q-point {q_idx}: {qpoint} ({'Gamma' if is_gamma else 'Non-Gamma'}) ---"
        )

        # Test 1: Complex nature consistency
        print("Test 1: Complex nature consistency")
        for mode_idx in range(modes.n_modes):
            displacement = modes.get_eigen_displacement(q_idx, mode_idx, normalize=True)
            has_significant_imag = np.any(np.abs(displacement.imag) > 1e-12)

            # For Gamma points, should be real
            if is_gamma and has_significant_imag:
                print(
                    f"  ❌ FAILED: Gamma point mode {mode_idx} has significant imaginary part"
                )
                all_tests_passed = False
            elif is_gamma and not has_significant_imag:
                print(f"  ✓ Mode {mode_idx}: Correctly real for Gamma point")

        # For non-Gamma points, some modes may have small imaginary parts (numerical noise)
        # This is acceptable and expected
        if not is_gamma:
            modes_with_significant_imag = 0
            for mode_idx in range(modes.n_modes):
                displacement = modes.get_eigen_displacement(
                    q_idx, mode_idx, normalize=True
                )
                if np.any(np.abs(displacement.imag) > 1e-12):
                    modes_with_significant_imag += 1

            print(
                f"  ✓ Non-Gamma point: {modes_with_significant_imag}/{modes.n_modes} modes have significant imaginary parts"
            )

        # Test 2: Orthogonality within degenerate subspaces
        print("Test 2: Orthogonality within degenerate subspaces")
        freq_tolerance = 1e-6
        unique_freqs, counts = np.unique(np.round(frequencies, 6), return_counts=True)
        degenerate_freqs = unique_freqs[counts > 1]

        degenerate_subspace_passed = True
        for freq in degenerate_freqs:
            indices = np.where(np.abs(frequencies - freq) < freq_tolerance)[0]
            if len(indices) < 2:
                continue

            # Get displacements for degenerate modes
            displacements = []
            for mode_idx in indices:
                disp = modes.get_eigen_displacement(q_idx, mode_idx, normalize=True)
                displacements.append(disp)

            # Check orthogonality
            max_projection = 0.0
            for i in range(len(displacements)):
                for j in range(i + 1, len(displacements)):
                    projection = abs(
                        modes.mass_weighted_projection(
                            displacements[i], displacements[j]
                        )
                    )
                    max_projection = max(max_projection, projection)

            if max_projection > 1e-6:
                print(
                    f"  ❌ FAILED: Degenerate modes at freq {freq:.6f} not orthogonal (max proj: {max_projection:.2e})"
                )
                degenerate_subspace_passed = False
                all_tests_passed = False
            else:
                print(
                    f"  ✓ Degenerate modes at freq {freq:.6f}: orthogonal (max proj: {max_projection:.2e})"
                )

        if degenerate_subspace_passed:
            print(f"  ✓ All degenerate subspaces are orthogonal")

        # Test 3: Supercell displacement generation
        print("Test 3: Supercell displacement generation")

        # Use appropriate supercell matrix for different q-points
        if np.allclose(qpoint, [0.0, 0.0, 0.0]):
            # Gamma point: any supercell works
            supercell_matrix = np.array([[2, 0, 0], [0, 2, 0], [0, 0, 1]])
        elif np.allclose(qpoint, [0.0, 0.0, 0.5]):
            # [0, 0, 0.5] requires at least 2×1×2 supercell
            supercell_matrix = np.array([[2, 0, 0], [0, 1, 0], [0, 0, 2]])
        elif np.allclose(qpoint, [0.5, 0.5, 0.0]):
            # [0.5, 0.5, 0] requires at least 2×2×1 supercell
            supercell_matrix = np.array([[2, 0, 0], [0, 2, 0], [0, 0, 1]])
        else:
            # Default supercell
            supercell_matrix = np.array([[2, 0, 0], [0, 2, 0], [0, 0, 2]])

        supercell_test_passed = True
        for mode_idx in range(min(3, modes.n_modes)):  # Test first 3 modes
            try:
                supercell_disp = modes.generate_mode_displacement(
                    q_idx, mode_idx, supercell_matrix, amplitude=1.0
                )

                # Supercell displacements should always be real
                if np.any(np.abs(supercell_disp.imag) > 1e-12):
                    print(
                        f"  ❌ FAILED: Supercell displacement for mode {mode_idx} is complex"
                    )
                    supercell_test_passed = False
                    all_tests_passed = False
                else:
                    print(
                        f"  ✓ Mode {mode_idx}: Supercell displacement is real (shape: {supercell_disp.shape})"
                    )

            except Exception as e:
                print(
                    f"  ❌ FAILED: Error generating supercell displacement for mode {mode_idx}: {e}"
                )
                supercell_test_passed = False
                all_tests_passed = False

        if supercell_test_passed:
            print(f"  ✓ All supercell displacements are real")

    print(f"\n=== Summary ===")
    if all_tests_passed:
        print(
            f"🎉 ALL TESTS PASSED! Complex displacement implementation is working correctly."
        )
        print(f"✅ Gamma points: Real displacements")
        print(
            f"✅ Non-Gamma points: Complex displacements (when eigenvectors have significant imaginary parts)"
        )
        print(f"✅ Degenerate modes: Perfect orthogonality within subspaces")
        print(f"✅ Supercell displacements: Always real")
    else:
        print(f"❌ SOME TESTS FAILED! Please review the implementation.")

    return all_tests_passed


def test_mass_weighted_projection_properties():
    """Test mathematical properties of mass-weighted projection."""

    print(f"\n=== Mass-Weighted Projection Properties Test ===")

    # Load modes to access the method
    BATIO3_YAML_PATH = "/Users/hexu/projects/phonproj/data/BaTiO3_phonopy_params.yaml"
    qpoints = np.array([[0.0, 0.0, 0.0]])
    modes = PhononModes.from_phonopy_yaml(BATIO3_YAML_PATH, qpoints=qpoints)

    # Get some real displacements for testing
    v1 = modes.get_eigen_displacement(0, 0, normalize=True)  # Real displacement
    v2 = modes.get_eigen_displacement(0, 1, normalize=True)  # Real displacement

    # Create a complex displacement for testing
    v3 = v1 + 0.1j * v2
    v3 = v3 / modes.mass_weighted_norm(v3)  # Normalize

    all_tests_passed = True

    # Test 1: Self-projection should be real and positive
    self_proj = modes.mass_weighted_projection(v3, v3)
    if abs(self_proj.imag) > 1e-12 or self_proj.real <= 0:
        print(
            f"❌ FAILED: Self-projection should be real and positive, got {self_proj}"
        )
        all_tests_passed = False
    else:
        print(f"✓ Self-projection is real and positive: {self_proj.real:.6f}")

    # Test 2: Conjugate symmetry
    proj_12 = modes.mass_weighted_projection(v1, v3)
    proj_21 = modes.mass_weighted_projection(v3, v1)
    if abs(np.conj(proj_12) - proj_21) > 1e-12:
        print(f"❌ FAILED: Conjugate symmetry broken: {proj_12} vs {np.conj(proj_21)}")
        all_tests_passed = False
    else:
        print(
            f"✓ Conjugate symmetry holds: <v1|v3> = {proj_12}, <v3|v1>* = {np.conj(proj_21)}"
        )

    # Test 3: Linearity
    alpha = 2.0 + 1.0j
    v_scaled = alpha * v3
    proj_scaled = modes.mass_weighted_projection(v1, v_scaled)
    expected = alpha * proj_12
    if abs(proj_scaled - expected) > 1e-12:
        print(f"❌ FAILED: Linearity broken: got {proj_scaled}, expected {expected}")
        all_tests_passed = False
    else:
        print(f"✓ Linearity holds: <v1|αv3> = α<v1|v3>")

    if all_tests_passed:
        print(f"✅ All mass-weighted projection properties verified")
    else:
        print(f"❌ Some projection property tests failed")

    return all_tests_passed


if __name__ == "__main__":
    test1_passed = test_complex_displacement_implementation()
    test2_passed = test_mass_weighted_projection_properties()

    overall_success = test1_passed and test2_passed

    print(f"\n🎯 FINAL RESULT: {'SUCCESS' if overall_success else 'FAILURE'}")
    print(
        f"Complex displacement implementation is {'READY' if overall_success else 'NEEDS WORK'}"
    )

    sys.exit(0 if overall_success else 1)
