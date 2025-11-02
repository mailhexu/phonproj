"""
Test Step 9: Complete Mode Decomposition

Tests for decomposing arbitrary displacements into contributions from all phonon modes
across all commensurate q-points in a supercell.

Note: BaTiO3 test dataset only contains Gamma point data, so decomposition completeness
depends on the nature of the displacement and available q-point coverage.
"""

import pytest
import numpy as np
from pathlib import Path

from phonproj.modes import PhononModes
from phonproj.core.structure_analysis import (
    decompose_displacement_to_modes,
    print_decomposition_table,
)


@pytest.fixture
def batio3_modes():
    """Load BaTiO3 phonon modes for testing."""
    yaml_path = "/Users/hexu/projects/phonproj/data/BaTiO3_phonopy_params.yaml"

    # Load only Gamma point (only available in this dataset)
    gamma_qpoint = np.array([[0.0, 0.0, 0.0]])
    modes = PhononModes.from_phonopy_yaml(yaml_path, gamma_qpoint)

    return modes


def test_decompose_single_mode_displacement_1x1x1(batio3_modes):
    """Test decomposition of single mode displacement in 1x1x1 supercell."""
    supercell_matrix = np.eye(3, dtype=int)

    # Generate displacement from a known mode (should decompose perfectly)
    test_displacement = batio3_modes.generate_mode_displacement(
        q_index=0, mode_index=14, supercell_matrix=supercell_matrix, amplitude=1.0
    )

    # Decompose displacement
    projection_table, summary = decompose_displacement_to_modes(
        displacement=test_displacement,
        phonon_modes=batio3_modes,
        supercell_matrix=supercell_matrix,
    )

    # Basic checks
    assert len(projection_table) > 0, "Should have projection data"
    assert summary["n_modes_total"] == 15, "Should use all 15 available modes"
    assert summary["n_qpoints"] == 1, "1x1x1 supercell should have 1 q-point (Gamma)"

    # Self-decomposition should be nearly perfect
    assert abs(summary["sum_squared_projections"] - 1.0) < 0.1, (
        f"Self-decomposition completeness: {summary['sum_squared_projections']:.6f}"
    )

    # The original mode should have the highest contribution
    mode_14_coeff = next(
        (
            entry["squared_coefficient"]
            for entry in projection_table
            if entry["mode_index"] == 14
        ),
        0,
    )
    assert mode_14_coeff > 0.9, f"Mode 14 should dominate with {mode_14_coeff:.6f}"


def test_decompose_random_displacement_1x1x1(batio3_modes):
    """Test decomposition of random displacement in 1x1x1 supercell."""
    supercell_matrix = np.eye(3, dtype=int)

    # Create random displacement
    np.random.seed(42)  # For reproducibility
    displacement = np.random.random((5, 3)) - 0.5  # Random between -0.5 and 0.5

    # Decompose displacement
    projection_table, summary = decompose_displacement_to_modes(
        displacement=displacement,
        phonon_modes=batio3_modes,
        supercell_matrix=supercell_matrix,
    )

    # Completeness depends on how well Gamma point modes span the displacement
    # This should be reasonably complete for 1x1x1
    assert summary["sum_squared_projections"] > 1.0, (
        f"Random displacement completeness: {summary['sum_squared_projections']:.6f}"
    )

    # Check that we get contributions from multiple modes
    significant_modes = [
        e for e in projection_table if abs(e["projection_coefficient"]) > 0.1
    ]
    assert len(significant_modes) > 3, (
        "Should have multiple significant mode contributions"
    )


def test_decompose_random_displacement_2x2x2(batio3_modes):
    """Test decomposition of random displacement in 2x2x2 supercell."""
    supercell_matrix = np.eye(3, dtype=int) * 2

    # Create random displacement for 2x2x2 supercell (5 atoms/cell × 8 cells = 40 atoms)
    np.random.seed(123)
    displacement = np.random.random((40, 3)) - 0.5

    # Decompose displacement - should raise ValueError due to missing q-points
    with pytest.raises(ValueError, match="Missing commensurate q-points"):
        decompose_displacement_to_modes(
            displacement=displacement,
            phonon_modes=batio3_modes,
            supercell_matrix=supercell_matrix,
        )


def test_decompose_single_mode_displacement_2x2x2(batio3_modes):
    """Test decomposition of single mode displacement in 2x2x2 supercell."""
    supercell_matrix = np.eye(3, dtype=int) * 2

    # Generate displacement from Gamma point mode (should still decompose well)
    test_displacement = batio3_modes.generate_mode_displacement(
        q_index=0, mode_index=14, supercell_matrix=supercell_matrix, amplitude=1.0
    )

    # Decompose displacement - should raise ValueError due to missing q-points
    with pytest.raises(ValueError, match="Missing commensurate q-points"):
        decompose_displacement_to_modes(
            displacement=test_displacement,
            phonon_modes=batio3_modes,
            supercell_matrix=supercell_matrix,
        )


def test_commensurate_qpoint_detection(batio3_modes):
    """Test that commensurate q-point detection works correctly."""

    # Test different supercell sizes
    test_cases = [
        (np.eye(3, dtype=int), 1),  # 1x1x1 -> 1 q-point
        (np.eye(3, dtype=int) * 2, 1),  # 2x2x2 -> 1 q-point (only Gamma available)
        (np.eye(3, dtype=int) * 3, 1),  # 3x3x3 -> 1 q-point (only Gamma available)
    ]

    for supercell_matrix, expected_qpoints in test_cases:
        commensurate_indices = batio3_modes.get_commensurate_qpoints(supercell_matrix)
        assert len(commensurate_indices) == expected_qpoints, (
            f"Supercell {np.diag(supercell_matrix)} should have {expected_qpoints} q-points, "
            f"got {len(commensurate_indices)}"
        )


def test_mixed_mode_decomposition(batio3_modes):
    """Test decomposition of displacement from mixed modes."""
    supercell_matrix = np.eye(3, dtype=int) * 2

    # Create displacement from multiple modes
    mixed_displacement = np.zeros((40, 3))
    input_modes = [5, 10, 14]

    for mode_idx in input_modes:
        mode_disp = batio3_modes.generate_mode_displacement(
            q_index=0,
            mode_index=mode_idx,
            supercell_matrix=supercell_matrix,
            amplitude=0.5,
        )
        mixed_displacement += mode_disp

    # Decompose - should raise ValueError due to missing q-points
    with pytest.raises(ValueError, match="Missing commensurate q-points"):
        decompose_displacement_to_modes(
            displacement=mixed_displacement,
            phonon_modes=batio3_modes,
            supercell_matrix=supercell_matrix,
        )


def test_missing_qpoints_error(batio3_modes):
    """Test that ValueError is raised when required q-points are missing."""
    # Use a supercell that requires q-points not in batio3_modes (which only has Gamma)
    supercell_matrix = (
        np.eye(3, dtype=int) * 2
    )  # 2x2x2 requires 8 q-points, but only Gamma is loaded

    # Create a test displacement
    test_displacement = np.random.random((40, 3))  # 2x2x2 supercell has 8*5=40 atoms

    # decompose_displacement should raise ValueError due to missing q-points
    with pytest.raises(ValueError, match="Missing commensurate q-points"):
        batio3_modes.decompose_displacement(
            displacement=test_displacement,
            supercell_matrix=supercell_matrix,
            normalize=True,
        )
