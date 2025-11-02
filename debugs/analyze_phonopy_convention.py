#!/usr/bin/env python3
"""
Analyze phonopy displacements to understand the phase convention.
"""

import numpy as np
from pathlib import Path
from phonproj.modes import PhononModes

# Data path
BATIO3_YAML_PATH = Path(__file__).parent / "data" / "BaTiO3_phonopy_params.yaml"


def analyze_phonopy_convention():
    """Try to reverse engineer phonopy's phase convention."""

    # Set up test case
    qpoints_2x2x2 = []
    for i in range(2):
        for j in range(2):
            for k in range(2):
                qpoints_2x2x2.append([i / 2.0, j / 2.0, k / 2.0])
    qpoints_2x2x2 = np.array(qpoints_2x2x2)

    modes = PhononModes.from_phonopy_yaml(str(BATIO3_YAML_PATH), qpoints_2x2x2)

    # Use 2x2x2 supercell
    supercell_matrix = np.eye(3, dtype=int) * 2

    # Find a non-Gamma q-point
    non_gamma_index = None
    for i, qpoint in enumerate(modes.qpoints):
        if not np.allclose(qpoint, [0, 0, 0], atol=1e-6):
            non_gamma_index = i
            break

    print(f"Q-point: {modes.qpoints[non_gamma_index]}")

    # Get phonopy displacement
    phonopy_displacement = modes._calculate_supercell_displacements_phonopy(
        non_gamma_index, 0, supercell_matrix, amplitude=1.0
    )

    # Normalize it
    current_norm = modes.mass_weighted_norm(phonopy_displacement)
    if current_norm > 1e-12:
        phonopy_displacement = phonopy_displacement * 1.0 / current_norm

    print(
        f"Phonopy displacement norm: {modes.mass_weighted_norm(phonopy_displacement)}"
    )

    # Get our R gauge displacement
    R_displacement = modes._calculate_supercell_displacements_direct(
        non_gamma_index, 0, supercell_matrix, amplitude=1.0, gauge="R"
    )

    print(f"R gauge displacement norm: {modes.mass_weighted_norm(R_displacement)}")

    # Look at the pattern of displacements
    print(f"\n=== DISPLACEMENT PATTERNS ===")
    print(f"Phonopy (first few atoms):")
    for i in range(min(8, len(phonopy_displacement))):
        print(f"  Atom {i}: {phonopy_displacement[i]}")

    print(f"\nR gauge (first few atoms):")
    for i in range(min(8, len(R_displacement))):
        print(f"  Atom {i}: {R_displacement[i]}")

    # Check if there's a simple relationship
    print(f"\n=== RELATIONSHIP ANALYSIS ===")
    correlation = np.dot(phonopy_displacement.flatten(), R_displacement.flatten()) / (
        np.linalg.norm(phonopy_displacement) * np.linalg.norm(R_displacement)
    )
    print(f"Correlation: {correlation}")

    # Check if phonopy displacement might be conjugate of R gauge
    R_conj_correlation = np.dot(
        phonopy_displacement.flatten(), np.conj(R_displacement.flatten())
    ) / (np.linalg.norm(phonopy_displacement) * np.linalg.norm(R_displacement))
    print(f"Correlation with conjugate: {R_conj_correlation}")

    # Check phase difference
    ratio = phonopy_displacement.flatten() / (R_displacement.flatten() + 1e-12)
    unique_ratios = []
    for r in ratio:
        if abs(r) > 1e-6:
            unique_ratios.append(r)

    if unique_ratios:
        print(f"Sample ratios (phonopy/R): {unique_ratios[:5]}")
        # Check if ratios have consistent phase
        phases = np.angle(unique_ratios)
        print(f"Phase angles: {phases[:5]}")
        print(f"Phase spread: {np.std(phases)}")

    # Try to see if there's a simple transformation
    print(f"\n=== TESTING SIMPLE PHASE TRANSFORMS ===")
    # Test multiplication by i
    i_transform = R_displacement * 1j
    i_correlation = np.dot(
        phonopy_displacement.flatten(), i_transform.flatten().real
    ) / (np.linalg.norm(phonopy_displacement) * np.linalg.norm(i_transform.real))
    print(f"Correlation with R*i (real part): {i_correlation}")

    # Test multiplication by -i
    minus_i_transform = R_displacement * (-1j)
    minus_i_correlation = np.dot(
        phonopy_displacement.flatten(), minus_i_transform.flatten().real
    ) / (np.linalg.norm(phonopy_displacement) * np.linalg.norm(minus_i_transform.real))
    print(f"Correlation with R*(-i) (real part): {minus_i_correlation}")


if __name__ == "__main__":
    analyze_phonopy_convention()
