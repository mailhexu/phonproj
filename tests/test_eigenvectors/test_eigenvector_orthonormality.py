"""
Test eigenvector orthonormality for phonon calculations.

This module tests that eigenvectors at a given q-point form an orthonormal basis,
which is a fundamental property of the dynamical matrix eigenvalue problem.

Uses the package method check_eigenvector_orthonormality() for verification.
"""

import pytest
from phonproj.band_structure import PhononBand


# Test data: (name, data_source, q_index, description)
TEST_CASES = [
    (
        "BaTiO3_gamma",
        "/Users/hexu/projects/phonproj/data/BaTiO3_phonopy_params.yaml",
        0,
        "Γ-point",
    ),
    (
        "BaTiO3_M_point",
        "/Users/hexu/projects/phonproj/data/BaTiO3_phonopy_params.yaml",
        "M",
        "M-point",
    ),
    (
        "PbTiO3_gamma",
        "/Users/hexu/projects/phonproj/data/yajundata/0.02-P4mmm-PTO",
        0,
        "Γ-point",
    ),
]


def load_band_structure(data_source: str) -> PhononBand:
    """Load band structure from data source."""
    if "yajundata" in data_source:
        # PbTiO3 data (may fail if forces missing)
        try:
            return PhononBand.calculate_band_structure_from_phonopy(
                data_source, path="GMXMG", npoints=50, units="cm-1"
            )
        except RuntimeError as e:
            if "missing forces" in str(e) or "not prepared" in str(e):
                pytest.skip(f"Forces not available in dataset: {e}")
            raise
    else:
        # BaTiO3 data
        return PhononBand.calculate_band_structure_from_phonopy(
            data_source, path="GMXMG", npoints=50, units="cm-1"
        )


def get_q_index(band: PhononBand, q_index: int | str) -> int:
    """Get actual q-point index from label or number."""
    if isinstance(q_index, str):
        if q_index == "M":
            # Find M-point from k-path labels
            m_indices = [
                idx for idx, label in band.kpath_data["kpath_labels"] if label == "M"
            ]
            return m_indices[1] if len(m_indices) >= 2 else 25
        else:
            raise ValueError(f"Unknown q-point label: {q_index}")
    return int(q_index)


@pytest.mark.parametrize(
    "name,data_source,q_index,description", TEST_CASES, ids=[x[0] for x in TEST_CASES]
)
def test_eigenvector_orthonormality(
    name: str, data_source: str, q_index: int | str, description: str
) -> None:
    """
    Test eigenvector orthonormality at various q-points.

    Args:
        name: Test case name
        data_source: Path to phonon data
        q_index: Q-point index or label ('M' for M-point)
        description: Human-readable description
    """
    # Load band structure
    band = load_band_structure(data_source)

    # Get actual q-point index
    actual_q_index = get_q_index(band, q_index)

    # Check orthonormality using package method
    tolerance = 1e-10
    is_orthonormal, max_error, errors = band.check_eigenvector_orthonormality(
        actual_q_index, tolerance=tolerance
    )

    # Verify results
    assert is_orthonormal, (
        f"Eigenvectors not orthonormal for {name} at {description}!\n"
        f"  Maximum deviation: {max_error:.2e}\n"
        f"  Tolerance: {tolerance:.0e}\n"
        f"  Details: {errors}"
    )

    # Verify excellent numerical accuracy
    assert max_error < 1e-14, (
        f"Numerical accuracy degraded for {name}!\n"
        f"  Expected: < 1e-14\n"
        f"  Got: {max_error:.2e}"
    )


def test_eigenvector_orthonormality_summary() -> None:
    """
    Summary test that verifies orthonormality for multiple q-points in one go.
    """
    # Load BaTiO3 data
    yaml_path = "/Users/hexu/projects/phonproj/data/BaTiO3_phonopy_params.yaml"
    band = PhononBand.calculate_band_structure_from_phonopy(
        yaml_path, path="GMXMG", npoints=50, units="cm-1"
    )

    # Test at multiple q-points
    q_points_to_test = [0, 10, 20, 30]  # Γ, mid-path, etc.
    tolerance = 1e-10

    print(
        f"\nTesting eigenvector orthonormality at {len(q_points_to_test)} q-points..."
    )

    for q_idx in q_points_to_test:
        if q_idx < band.frequencies.shape[0]:
            is_ortho, max_err, _ = band.check_eigenvector_orthonormality(
                q_idx, tolerance=tolerance
            )
            assert is_ortho, f"Failed at q-point {q_idx}: error={max_err:.2e}"
            print(f"  ✓ q-point {q_idx:2d}: max_error={max_err:.2e}")

    print(f"\n✓ All {len(q_points_to_test)} q-points passed orthonormality check!")


import numpy as np
from phonproj.modes import PhononModes


def test_eigenvector_orthonormality_arbitrary_qpoints() -> None:
    """
    Test eigenvector orthonormality at arbitrary q-points using PhononModes.
    """
    yaml_path = "/Users/hexu/projects/phonproj/data/BaTiO3_phonopy_params.yaml"
    # Define arbitrary q-points (not just band path)
    qpoints = np.array(
        [
            [0.0, 0.0, 0.0],  # Gamma
            [0.5, 0.0, 0.0],  # X
            [0.5, 0.5, 0.0],  # M
            [0.5, 0.5, 0.5],  # R
            [0.33, 0.25, 0.1],  # Arbitrary
        ]
    )
    modes = PhononModes.from_phonopy_yaml(yaml_path, qpoints)
    tolerance = 1e-10
    for q_idx in range(modes.n_qpoints):
        is_ortho, max_err, _ = modes.check_eigenvector_orthonormality(
            q_idx, tolerance=tolerance
        )
        assert is_ortho, f"Failed at q-point {qpoints[q_idx]}: error={max_err:.2e}"
        assert max_err < 1e-14, (
            f"Numerical accuracy degraded at q-point {qpoints[q_idx]}! Got: {max_err:.2e}"
        )


if __name__ == "__main__":  # type: ignore
    # Run all test cases manually
    print("\n" + "=" * 70)
    print("Running Eigenvector Orthonormality Tests")
    print("=" * 70)

    for name, data_source, q_index, description in TEST_CASES:
        try:
            band = load_band_structure(data_source)
            actual_q_index = get_q_index(band, q_index)

            print(f"\nTest: {name} ({description})")
            is_ortho, max_err, errors = band.check_eigenvector_orthonormality(
                actual_q_index, tolerance=1e-10, verbose=True
            )

            if is_ortho:
                print(f"  ✓ PASS: Max error = {max_err:.2e}")
            else:
                print(f"  ✗ FAIL: Max error = {max_err:.2e}")

        except Exception as e:
            print(f"  ⚠ SKIP: {e}")

    print("\n" + "=" * 70)
    print("Running Summary Test")
    print("=" * 70)
    test_eigenvector_orthonormality_summary()

    print("\n" + "=" * 70)
    print("All tests completed!")
    print("=" * 70)
