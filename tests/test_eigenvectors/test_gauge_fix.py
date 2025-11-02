#!/usr/bin/env python3
"""
Simple test to verify the gauge fix is working correctly.
"""

import numpy as np
from pathlib import Path
from phonproj.modes import PhononModes


def test_gauge_fix():
    """Test that R gauge doesn't apply phase factors."""

    # Load BaTiO3 data
    BATIO3_YAML_PATH = Path("data/BaTiO3_phonopy_params.yaml")

    # Use just Gamma point for simplicity
    qpoints = np.array([[0.0, 0.0, 0.0]])
    modes = PhononModes.from_phonopy_yaml(str(BATIO3_YAML_PATH), qpoints)

    print(f"Gauge: {modes.gauge}")

    # Use 2x2x2 supercell
    supercell_matrix = np.eye(3, dtype=int) * 2

    # Get displacement for first mode at Gamma
    qpoint_idx = 0
    mode_idx = 0

    displacement = modes.generate_mode_displacement(
        qpoint_idx, mode_idx, supercell_matrix, amplitude=1.0
    )

    print(f"Displacement shape: {displacement.shape}")
    print(f"Displacement is real: {np.allclose(displacement.imag, 0)}")
    print(f"Max imaginary component: {np.max(np.abs(displacement.imag))}")
    print(f"Displacement norm: {np.linalg.norm(displacement)}")

    # For R gauge at Gamma point, displacement should be purely real
    # and should match the eigenvector components
    eigenvector = modes.eigenvectors[qpoint_idx][mode_idx]
    frequency = modes.frequencies[qpoint_idx][mode_idx]

    print(f"Eigenvector shape: {eigenvector.shape}")
    print(f"Eigenvector is real: {np.allclose(eigenvector.imag, 0)}")
    print(f"Frequency: {frequency:.6f} THz")

    # Test with a non-Gamma point to see difference
    qpoints_with_k = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.5]])
    modes_with_k = PhononModes.from_phonopy_yaml(str(BATIO3_YAML_PATH), qpoints_with_k)

    # Get displacement for first mode at k=[0,0,0.5]
    displacement_k = modes_with_k.generate_mode_displacement(
        1,
        0,
        supercell_matrix,
        amplitude=1.0,  # qpoint_idx=1 corresponds to [0,0,0.5]
    )

    print(f"\nFor q=[0,0,0.5]:")
    print(f"Displacement is real: {np.allclose(displacement_k.imag, 0)}")
    print(f"Max imaginary component: {np.max(np.abs(displacement_k.imag))}")
    print(f"Displacement norm: {np.linalg.norm(displacement_k)}")

    # Check if displacement is zero (which would indicate cancellation due to phase factors)
    print(f"Displacement is zero: {np.allclose(displacement_k, 0, atol=1e-10)}")
    print(f"Max displacement component: {np.max(np.abs(displacement_k))}")


if __name__ == "__main__":
    test_gauge_fix()
