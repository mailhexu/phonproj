#!/usr/bin/env python3
"""
Test to verify our complex displacement implementation preserves degenerate mode orthogonality.
"""

import numpy as np
import sys

sys.path.insert(0, "/Users/hexu/projects/phonproj")

from phonproj.modes import PhononModes


def test_complex_displacement_orthogonality():
    """Test that our complex displacements preserve orthogonality within degenerate subspaces."""

    # Load test data with non-Gamma q-points
    BATIO3_YAML_PATH = "/Users/hexu/projects/phonproj/data/BaTiO3_phonopy_params.yaml"
    qpoints = np.array(
        [
            [0.0, 0.0, 0.0],  # Gamma point (control)
            [0.0, 0.0, 0.5],  # Non-Gamma point with degenerate modes
            [0.25, 0.25, 0.25],  # Another non-Gamma point
        ]
    )
    modes = PhononModes.from_phonopy_yaml(BATIO3_YAML_PATH, qpoints=qpoints)

    print("=== Testing Complex Displacement Orthogonality ===")
    print(f"Loaded {modes.n_qpoints} q-points with {modes.n_modes} modes")

    for q_idx in range(modes.n_qpoints):
        qpoint = modes.qpoints[q_idx]
        is_gamma = np.allclose(qpoint, 0.0, atol=1e-6)
        frequencies = modes.frequencies[q_idx]

        print(
            f"\n--- Q-point {q_idx}: {qpoint} ({'Gamma' if is_gamma else 'Non-Gamma'}) ---"
        )

        # Find degenerate modes
        freq_tolerance = 1e-6
        unique_freqs, counts = np.unique(np.round(frequencies, 6), return_counts=True)
        degenerate_freqs = unique_freqs[counts > 1]

        print(f"Found {len(degenerate_freqs)} degenerate frequencies")

        for freq in degenerate_freqs:
            indices = np.where(np.abs(frequencies - freq) < freq_tolerance)[0]
            if len(indices) < 2:
                continue

            print(
                f"\nTesting degenerate modes at frequency {freq:.6f}: modes {list(indices)}"
            )

            # Get displacements using our implementation
            displacements = []
            for mode_idx in indices:
                disp = modes.get_eigen_displacement(q_idx, mode_idx, normalize=True)
                displacements.append(disp)

                # Check if displacement is complex for non-Gamma
                has_imaginary = np.any(np.abs(disp.imag) > 1e-12)
                expected_complex = not is_gamma
                print(
                    f"  Mode {mode_idx}: complex={has_imaginary}, expected_complex={expected_complex}"
                )

                if has_imaginary != expected_complex:
                    print(f"    ⚠ WARNING: Complex nature doesn't match expectation!")

            # Test orthogonality within degenerate subspace
            print(f"  Testing orthogonality within degenerate subspace:")
            max_projection = 0.0
            for i in range(len(displacements)):
                for j in range(i + 1, len(displacements)):
                    # Use mass-weighted projection
                    projection = modes.mass_weighted_projection(
                        displacements[i], displacements[j]
                    )
                    abs_projection = abs(projection)
                    max_projection = max(max_projection, abs_projection)

                    mode_i = indices[i]
                    mode_j = indices[j]
                    print(
                        f"    Modes {mode_i}-{mode_j}: projection = {abs_projection:.8f}"
                    )

                    if abs_projection > 1e-6:
                        print(f"      ⚠ NON-ORTHOGONAL! Expected ~0")
                    else:
                        print(f"      ✓ Orthogonal")

            print(f"  Maximum projection in degenerate subspace: {max_projection:.8f}")

            if max_projection < 1e-6:
                print(f"  ✓ PASSED: Degenerate modes are orthogonal")
            else:
                print(f"  ❌ FAILED: Degenerate modes are not orthogonal")


def test_mass_weighted_projection_complex():
    """Test that our mass-weighted projection correctly handles complex numbers."""

    print(f"\n=== Testing Mass-Weighted Projection with Complex Numbers ===")

    # Load a real structure for realistic masses and correct dimensions
    from phonproj.core.io import create_phonopy_object, phonopy_to_ase
    from pathlib import Path

    phonopy = create_phonopy_object(
        Path("/Users/hexu/projects/phonproj/data/BaTiO3_phonopy_params.yaml")
    )
    primitive_cell = phonopy_to_ase(phonopy.primitive)
    n_atoms = len(primitive_cell)
    masses = primitive_cell.get_masses()

    print(f"Using BaTiO3 structure with {n_atoms} atoms")

    # Create simple test vectors with correct shape (n_atoms, 3)
    # Complex test vectors - only set non-zero values for first 3 atoms to keep it simple
    v1 = np.zeros((n_atoms, 3), dtype=complex)
    v1[0] = [1.0 + 2.0j, 0.0, 0.0]
    v1[1] = [0.0, 1.0, 0.0]
    v1[2] = [0.0, 0.0, 1.0 - 1.0j]

    v2 = np.zeros((n_atoms, 3), dtype=complex)
    v2[0] = [1.0 - 2.0j, 0.0, 0.0]
    v2[1] = [0.0, 1.0, 0.0]
    v2[2] = [0.0, 0.0, 1.0 + 1.0j]

    v3 = np.zeros((n_atoms, 3), dtype=complex)
    v3[0] = [0.0, 1.0, 0.0]
    v3[1] = [1.0, 0.0, 0.0]

    # Create a dummy PhononModes object to access the method
    # Use minimal data with correct dimensions
    qpoints = np.array([[0.0, 0.0, 0.0]])
    frequencies = np.array([[1.0, 2.0, 3.0]])
    eigenvectors = np.zeros((1, 3, n_atoms * 3), dtype=complex)

    modes = PhononModes(
        primitive_cell=primitive_cell,
        qpoints=qpoints,
        frequencies=frequencies,
        eigenvectors=eigenvectors,
        atomic_masses=masses,
        gauge="R",
    )

    # Test self-projection (should be real and positive)
    proj_self = modes.mass_weighted_projection(v1, v1)
    print(f"Self-projection v1·v1 = {proj_self}")
    print(f"  Real part: {proj_self.real:.6f}, Imaginary part: {proj_self.imag:.6f}")

    if abs(proj_self.imag) < 1e-12 and proj_self.real > 0:
        print(f"  ✓ PASSED: Self-projection is real and positive")
    else:
        print(f"  ❌ FAILED: Self-projection should be real and positive")

    # Test complex conjugate relationship
    proj_12 = modes.mass_weighted_projection(v1, v2)
    proj_21 = modes.mass_weighted_projection(v2, v1)
    print(f"\nConjugate test:")
    print(f"  v1·v2 = {proj_12}")
    print(f"  v2·v1 = {proj_21}")
    print(f"  Conjugate relationship: {np.conj(proj_12)} vs {proj_21}")

    if abs(np.conj(proj_12) - proj_21) < 1e-12:
        print(f"  ✓ PASSED: Conjugate relationship holds")
    else:
        print(f"  ❌ FAILED: Conjugate relationship broken")

    # Test orthogonal vectors
    proj_13 = modes.mass_weighted_projection(v1, v3)
    print(f"\nOrthogonal test:")
    print(f"  v1·v3 = {proj_13}")
    print(f"  |v1·v3| = {abs(proj_13):.8f}")

    if abs(proj_13) < 1e-12:
        print(f"  ✓ PASSED: Orthogonal vectors have zero projection")
    else:
        print(f"  ❌ FAILED: Should be orthogonal")


if __name__ == "__main__":
    test_complex_displacement_orthogonality()
    test_mass_weighted_projection_complex()
