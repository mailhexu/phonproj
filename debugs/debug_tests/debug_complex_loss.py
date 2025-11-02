#!/usr/bin/env python3
"""
Debug why some non-Gamma modes are becoming real instead of staying complex.
"""

import numpy as np
import sys

sys.path.insert(0, "/Users/hexu/projects/phonproj")

from phonproj.modes import PhononModes


def debug_complex_loss():
    """Debug why some complex modes are becoming real."""

    # Load test data
    BATIO3_YAML_PATH = "/Users/hexu/projects/phonproj/data/BaTiO3_phonopy_params.yaml"
    qpoints = np.array([[0.0, 0.0, 0.5]])  # Non-Gamma point with the issue
    modes = PhononModes.from_phonopy_yaml(BATIO3_YAML_PATH, qpoints=qpoints)

    print("=== Debugging Complex Loss ===")

    q_idx = 0
    qpoint = modes.qpoints[q_idx]
    frequencies = modes.frequencies[q_idx]
    eigenvectors = modes.eigenvectors[q_idx]

    print(f"Q-point: {qpoint}")
    print(f"Raw eigenvectors shape: {eigenvectors.shape}")
    print(f"Raw eigenvectors dtype: {eigenvectors.dtype}")

    # Look at specific problematic modes
    problematic_modes = [
        0,
        1,
        5,
        6,
        11,
        12,
    ]  # These became real when they should be complex
    working_modes = [2, 3, 9, 10]  # These stayed complex as expected

    print(f"\n--- Analyzing Problematic Modes (became real) ---")
    for mode_idx in problematic_modes:
        print(f"\nMode {mode_idx} (freq = {frequencies[mode_idx]:.6f}):")

        # Raw eigenvector
        raw_eigvec = eigenvectors[mode_idx]
        raw_reshaped = raw_eigvec.reshape(modes.n_atoms, 3)

        print(f"  Raw eigenvector max |imag|: {np.max(np.abs(raw_reshaped.imag)):.2e}")
        print(f"  Raw eigenvector max |real|: {np.max(np.abs(raw_reshaped.real)):.2e}")
        print(
            f"  Raw has significant imaginary: {np.any(np.abs(raw_reshaped.imag) > 1e-12)}"
        )

        # Mass-weighted displacement
        mass_weights = np.sqrt(modes.atomic_masses)
        mass_weighted = raw_reshaped / mass_weights[:, np.newaxis]

        print(f"  Mass-weighted max |imag|: {np.max(np.abs(mass_weighted.imag)):.2e}")
        print(f"  Mass-weighted max |real|: {np.max(np.abs(mass_weighted.real)):.2e}")
        print(
            f"  Mass-weighted has significant imaginary: {np.any(np.abs(mass_weighted.imag) > 1e-12)}"
        )

        # Normalized displacement
        norm = modes.mass_weighted_norm(mass_weighted)
        normalized = mass_weighted / norm if norm > 1e-12 else mass_weighted

        print(f"  Normalized max |imag|: {np.max(np.abs(normalized.imag)):.2e}")
        print(f"  Normalized max |real|: {np.max(np.abs(normalized.real)):.2e}")
        print(
            f"  Normalized has significant imaginary: {np.any(np.abs(normalized.imag) > 1e-12)}"
        )

        # What our method returns
        our_result = modes.get_eigen_displacement(q_idx, mode_idx, normalize=True)
        print(f"  Our result max |imag|: {np.max(np.abs(our_result.imag)):.2e}")
        print(f"  Our result max |real|: {np.max(np.abs(our_result.real)):.2e}")
        print(
            f"  Our result has significant imaginary: {np.any(np.abs(our_result.imag) > 1e-12)}"
        )

    print(f"\n--- Analyzing Working Modes (stayed complex) ---")
    for mode_idx in working_modes:
        print(f"\nMode {mode_idx} (freq = {frequencies[mode_idx]:.6f}):")

        # Raw eigenvector
        raw_eigvec = eigenvectors[mode_idx]
        raw_reshaped = raw_eigvec.reshape(modes.n_atoms, 3)

        print(f"  Raw eigenvector max |imag|: {np.max(np.abs(raw_reshaped.imag)):.2e}")
        print(f"  Raw eigenvector max |real|: {np.max(np.abs(raw_reshaped.real)):.2e}")
        print(
            f"  Raw has significant imaginary: {np.any(np.abs(raw_reshaped.imag) > 1e-12)}"
        )

        # What our method returns
        our_result = modes.get_eigen_displacement(q_idx, mode_idx, normalize=True)
        print(f"  Our result max |imag|: {np.max(np.abs(our_result.imag)):.2e}")
        print(f"  Our result max |real|: {np.max(np.abs(our_result.real)):.2e}")
        print(
            f"  Our result has significant imaginary: {np.any(np.abs(our_result.imag) > 1e-12)}"
        )


def test_mass_weighted_norm_complex():
    """Test if mass_weighted_norm preserves complex nature."""

    print(f"\n=== Testing Mass-Weighted Norm with Complex Numbers ===")

    # Load modes to access the method
    BATIO3_YAML_PATH = "/Users/hexu/projects/phonproj/data/BaTiO3_phonopy_params.yaml"
    qpoints = np.array([[0.0, 0.0, 0.0]])
    modes = PhononModes.from_phonopy_yaml(BATIO3_YAML_PATH, qpoints=qpoints)

    # Create test vectors
    # Complex vector with small imaginary part
    v_small_imag = np.array([[1.0 + 1e-8j, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])

    # Complex vector with large imaginary part
    v_large_imag = np.array([[1.0 + 0.5j, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])

    # Test norms
    norm_small = modes.mass_weighted_norm(v_small_imag)
    norm_large = modes.mass_weighted_norm(v_large_imag)

    print(f"Small imaginary part vector norm: {norm_small}")
    print(f"Large imaginary part vector norm: {norm_large}")

    # Test normalization
    v_small_normalized = v_small_imag / norm_small
    v_large_normalized = v_large_imag / norm_large

    print(
        f"Small imag normalized max |imag|: {np.max(np.abs(v_small_normalized.imag)):.2e}"
    )
    print(
        f"Large imag normalized max |imag|: {np.max(np.abs(v_large_normalized.imag)):.2e}"
    )

    print(
        f"Small imag has significant imaginary: {np.any(np.abs(v_small_normalized.imag) > 1e-12)}"
    )
    print(
        f"Large imag has significant imaginary: {np.any(np.abs(v_large_normalized.imag) > 1e-12)}"
    )


if __name__ == "__main__":
    debug_complex_loss()
    test_mass_weighted_norm_complex()
