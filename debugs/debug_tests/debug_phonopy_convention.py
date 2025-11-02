#!/usr/bin/env python3
"""
Debug script to investigate phonopy eigenvector conventions and orthogonality.

This script examines:
1. Whether phonopy eigenvectors are orthogonal in their raw form
2. How mass-weighting affects orthogonality
3. Different normalization conventions
"""

import numpy as np
import sys
import os

# Add the project root to Python path
sys.path.insert(0, "/Users/hexu/projects/phonproj")

from phonproj.modes import PhononModes


def mass_weighted_dot(v1, v2, masses):
    """Calculate mass-weighted dot product between two eigenvectors"""
    # v1 and v2 are flat eigenvectors (3*n_atoms,)
    n_atoms = len(masses)
    v1_reshaped = v1.reshape(n_atoms, 3)
    v2_reshaped = v2.reshape(n_atoms, 3)

    # Mass-weighted dot product: sum over all components of v1_i * v2_i / m_i
    mass_weights = np.sqrt(masses)
    v1_weighted = v1_reshaped / mass_weights[:, np.newaxis]
    v2_weighted = v2_reshaped / mass_weights[:, np.newaxis]

    return np.sum(np.conj(v1_weighted) * v2_weighted)


def check_orthogonality(eigenvectors, masses, freq_tolerance=1e-6):
    """Check orthogonality of eigenvectors"""
    n_modes = eigenvectors.shape[0]

    print(f"Checking orthogonality for {n_modes} modes...")

    # Check all pairs
    max_projection = 0.0
    max_indices = (0, 0)

    for i in range(n_modes):
        for j in range(i + 1, n_modes):
            # Raw dot product (no mass weighting)
            raw_dot = np.conj(eigenvectors[i]) @ eigenvectors[j]

            # Mass-weighted dot product
            mass_dot = mass_weighted_dot(eigenvectors[i], eigenvectors[j], masses)

            abs_raw_dot = abs(raw_dot)
            abs_mass_dot = abs(mass_dot)

            if abs_mass_dot > max_projection:
                max_projection = abs_mass_dot
                max_indices = (i, j)

            # Print non-orthogonal pairs
            if abs_mass_dot > 1e-6:
                print(
                    f"  Modes {i:2d}-{j:2d}: raw_dot={abs_raw_dot:.6f}, mass_weighted_dot={abs_mass_dot:.6f}"
                )

    print(
        f"Maximum mass-weighted projection: {max_projection:.6f} between modes {max_indices}"
    )
    return max_projection


def analyze_phonopy_eigenvectors():
    """Analyze phonopy eigenvector conventions"""

    # Load test system (BaTiO3)
    yaml_file = "/Users/hexu/projects/phonproj/data/BaTiO3_phonopy_params.yaml"
    if not os.path.exists(yaml_file):
        print(f"Test file not found: {yaml_file}")
        return

    # Define some test q-points
    qpoints = np.array(
        [
            [0.0, 0.0, 0.0],  # Gamma point
            [0.0, 0.0, 0.5],  # Non-Gamma point
            [0.25, 0.25, 0.25],  # Another non-Gamma point
        ]
    )

    modes = PhononModes.from_phonopy_yaml(yaml_file, qpoints)

    print("=== Phonopy Eigenvector Convention Analysis ===")
    print(f"System: {len(modes.atomic_masses)} atoms")
    print(f"Q-points: {len(modes.qpoints)}")
    print(f"Atomic masses: {modes.atomic_masses}")

    # Test different q-points
    for q_idx in range(min(3, len(modes.qpoints))):
        qpoint = modes.qpoints[q_idx]
        is_gamma = np.allclose(qpoint, 0.0, atol=1e-6)

        print(
            f"\n--- Q-point {q_idx}: {qpoint} ({'Gamma' if is_gamma else 'Non-Gamma'}) ---"
        )

        # Get raw eigenvectors
        eigenvectors = modes.eigenvectors[q_idx]  # Shape: (n_modes, 3*n_atoms)
        frequencies = modes.frequencies[q_idx]

        print(f"Eigenvector shape: {eigenvectors.shape}")
        print(f"Eigenvector dtype: {eigenvectors.dtype}")
        print(f"Frequencies: {frequencies[:6]} ...")

        # Check if eigenvectors are complex
        has_imaginary = np.any(np.abs(eigenvectors.imag) > 1e-12)
        print(f"Has significant imaginary parts: {has_imaginary}")

        # Check normalization of raw eigenvectors
        print("\nRaw eigenvector norms:")
        for i in range(min(6, len(eigenvectors))):
            norm = np.linalg.norm(eigenvectors[i])
            print(f"  Mode {i}: norm = {norm:.6f}")

        # Check orthogonality
        print("\nOrthogonality check:")
        max_proj = check_orthogonality(eigenvectors, modes.atomic_masses)

        # Find degenerate modes
        print(f"\nDegenerate modes (tolerance=1e-6):")
        unique_freqs, counts = np.unique(np.round(frequencies, 6), return_counts=True)
        degenerate_freqs = unique_freqs[counts > 1]

        for freq in degenerate_freqs:
            indices = np.where(np.abs(frequencies - freq) < 1e-6)[0]
            print(f"  Frequency {freq:.6f}: modes {list(indices)}")

            # Check orthogonality within degenerate subspace
            if len(indices) > 1:
                print(f"    Checking orthogonality within degenerate subspace:")
                degen_eigvecs = eigenvectors[indices]
                degen_max_proj = check_orthogonality(degen_eigvecs, modes.atomic_masses)


def test_simple_diatomic_chain():
    """Test with a simple diatomic chain to validate understanding"""
    print("\n=== Testing Simple Diatomic Chain ===")

    # Look for simpler test cases
    test_dir = "/Users/hexu/projects/phonproj/data"
    if os.path.exists(test_dir):
        print(f"Available test systems in {test_dir}:")
        for item in os.listdir(test_dir):
            item_path = os.path.join(test_dir, item)
            if os.path.isdir(item_path):
                print(f"  - {item}")
    else:
        print(f"Test directory not found: {test_dir}")


if __name__ == "__main__":
    analyze_phonopy_eigenvectors()
    test_simple_diatomic_chain()
