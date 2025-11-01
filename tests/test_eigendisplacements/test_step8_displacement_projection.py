"""
Tests for Step 8 functionality: Cross-supercell displacement projection.

Tests according to plan.md requirements:
- Test projection between identical supercells and displacements (normalized and unnormalized)
- Test projection with translated supercell (atom positions shifted due to periodic boundaries)
- Test projection with shuffled atoms and correspondingly shuffled displacements
- Test projection with combination of translation and shuffling
"""

import pytest
import numpy as np
from pathlib import Path
import copy

from phonproj.modes import PhononModes
from phonproj.core.structure_analysis import (
    project_displacements_between_supercells,
    create_atom_mapping,
)


# Test data paths
BATIO3_YAML_PATH = (
    Path(__file__).parent.parent.parent / "data" / "BaTiO3_phonopy_params.yaml"
)


def _extract_atoms(supercell_result):
    """Helper function to extract Atoms object from generate_displaced_supercell result."""
    if isinstance(supercell_result, tuple):
        return supercell_result[0]
    else:
        return supercell_result


class TestDisplacementProjection:
    """Test suite for Step 8 cross-supercell displacement projection functionality."""

    def test_identical_supercells_normalized(self):
        """
        Test projection between two identical supercells and displacements.
        Should return coefficient of 1.0 when normalized.
        """
        # Load BaTiO3 data
        gamma_qpoint = np.array([[0.0, 0.0, 0.0]])
        modes = PhononModes.from_phonopy_yaml(str(BATIO3_YAML_PATH), gamma_qpoint)

        # Use 2x2x2 supercell for this test
        supercell_matrix = np.eye(3, dtype=int) * 2

        # Generate a displacement for mode 0 at Gamma point
        displacement = modes.generate_mode_displacement(
            q_index=0, mode_index=6, supercell_matrix=supercell_matrix, amplitude=1.0
        )

        # Create identical supercells
        source_supercell_result = modes.generate_displaced_supercell(
            q_index=0,
            mode_index=6,
            supercell_matrix=supercell_matrix,
            amplitude=0.0,  # Undisplaced supercell
            return_displacements=False,
        )
        source_supercell = _extract_atoms(source_supercell_result)
        target_supercell = copy.deepcopy(source_supercell)

        # Project identical displacement onto itself (normalized)
        coeff = project_displacements_between_supercells(
            source_displacement=displacement,
            target_displacement=displacement,
            source_supercell=source_supercell,
            target_supercell=target_supercell,
            normalize=True,
        )

        # Should be 1.0 within tolerance
        assert abs(coeff - 1.0) < 1e-12, (
            f"Identical supercells normalized projection failed: coefficient = {coeff}"
        )

    def test_identical_supercells_unnormalized(self):
        """
        Test projection between two identical supercells and displacements.
        Should return the mass-weighted norm squared when unnormalized.
        """
        # Load BaTiO3 data
        gamma_qpoint = np.array([[0.0, 0.0, 0.0]])
        modes = PhononModes.from_phonopy_yaml(str(BATIO3_YAML_PATH), gamma_qpoint)

        # Use 2x2x2 supercell for this test
        supercell_matrix = np.eye(3, dtype=int) * 2

        # Generate a displacement for mode 0 at Gamma point
        displacement = modes.generate_mode_displacement(
            q_index=0, mode_index=6, supercell_matrix=supercell_matrix, amplitude=1.0
        )

        # Create identical supercells
        source_supercell_result = modes.generate_displaced_supercell(
            q_index=0,
            mode_index=6,
            supercell_matrix=supercell_matrix,
            amplitude=0.0,  # Undisplaced supercell
            return_displacements=False,
        )
        source_supercell = _extract_atoms(source_supercell_result)
        target_supercell = copy.deepcopy(source_supercell)

        # Project identical displacement onto itself (unnormalized)
        coeff_unnormalized = project_displacements_between_supercells(
            source_displacement=displacement,
            target_displacement=displacement,
            source_supercell=source_supercell,
            target_supercell=target_supercell,
            normalize=False,
        )

        # Calculate expected mass-weighted norm squared
        target_atomic_masses = target_supercell.get_masses()
        target_masses = np.repeat(target_atomic_masses, 3)
        displacement_flat = displacement.ravel()
        expected_norm_squared = np.sum(
            target_masses * displacement_flat.conj() * displacement_flat
        ).real

        # Should match within tolerance
        assert abs(coeff_unnormalized - expected_norm_squared) < 1e-10, (
            f"Identical supercells unnormalized projection failed: "
            f"coefficient = {coeff_unnormalized}, expected = {expected_norm_squared}"
        )

    def test_translated_supercell(self):
        """
        Test projection where one supercell is a translated version of the other.
        This tests handling of periodic boundary conditions.
        """
        # Load BaTiO3 data
        gamma_qpoint = np.array([[0.0, 0.0, 0.0]])
        modes = PhononModes.from_phonopy_yaml(str(BATIO3_YAML_PATH), gamma_qpoint)

        # Use 2x2x2 supercell
        supercell_matrix = np.eye(3, dtype=int) * 2

        # Generate a displacement
        displacement = modes.generate_mode_displacement(
            q_index=0, mode_index=6, supercell_matrix=supercell_matrix, amplitude=1.0
        )

        # Create source supercell
        source_supercell_result = modes.generate_displaced_supercell(
            q_index=0,
            mode_index=6,
            supercell_matrix=supercell_matrix,
            amplitude=0.0,
            return_displacements=False,
        )
        source_supercell = _extract_atoms(source_supercell_result)

        # Create translated target supercell by shifting all atomic positions
        target_supercell = copy.deepcopy(source_supercell)
        translation_vector = np.array([0.5, 0.3, 0.2])  # Arbitrary translation
        target_supercell.positions = target_supercell.positions + translation_vector

        # Apply periodic boundary conditions
        target_supercell.wrap()

        # Project displacement (should still work due to atom mapping)
        coeff = project_displacements_between_supercells(
            source_displacement=displacement,
            target_displacement=displacement,  # Same displacement pattern
            source_supercell=source_supercell,
            target_supercell=target_supercell,
            normalize=True,
        )

        # Should still be close to 1.0 (allowing for small numerical errors from translation)
        assert abs(coeff - 1.0) < 1e-6, (
            f"Translated supercell projection failed: coefficient = {coeff}"
        )

    def test_shuffled_atoms_and_displacements(self):
        """
        Test projection with shuffled atom ordering and correspondingly shuffled displacements.
        """
        # Load BaTiO3 data
        gamma_qpoint = np.array([[0.0, 0.0, 0.0]])
        modes = PhononModes.from_phonopy_yaml(str(BATIO3_YAML_PATH), gamma_qpoint)

        # Use 2x2x2 supercell
        supercell_matrix = np.eye(3, dtype=int) * 2

        # Generate a displacement
        source_displacement = modes.generate_mode_displacement(
            q_index=0, mode_index=6, supercell_matrix=supercell_matrix, amplitude=1.0
        )

        # Create source supercell
        source_supercell_result = modes.generate_displaced_supercell(
            q_index=0,
            mode_index=6,
            supercell_matrix=supercell_matrix,
            amplitude=0.0,
            return_displacements=False,
        )
        source_supercell = _extract_atoms(source_supercell_result)

        # Create shuffled target supercell and displacement
        np.random.seed(42)  # For reproducibility
        n_atoms = len(source_supercell)
        shuffle_indices = np.arange(n_atoms)
        np.random.shuffle(shuffle_indices)

        # Create shuffled supercell
        target_supercell = copy.deepcopy(source_supercell)
        target_supercell.positions = target_supercell.positions[shuffle_indices]
        target_supercell.numbers = target_supercell.numbers[shuffle_indices]

        # Create correspondingly shuffled displacement
        target_displacement = source_displacement[shuffle_indices]

        # Project displacement (should work with automatic atom mapping)
        coeff = project_displacements_between_supercells(
            source_displacement=source_displacement,
            target_displacement=target_displacement,
            source_supercell=source_supercell,
            target_supercell=target_supercell,
            normalize=True,
        )

        # Should be close to 1.0
        assert abs(coeff - 1.0) < 1e-6, (
            f"Shuffled atoms projection failed: coefficient = {coeff}"
        )

    def test_translation_and_shuffling_combined(self):
        """
        Test projection with combination of translation and atom shuffling.
        This is the most challenging scenario.
        """
        # Load BaTiO3 data
        gamma_qpoint = np.array([[0.0, 0.0, 0.0]])
        modes = PhononModes.from_phonopy_yaml(str(BATIO3_YAML_PATH), gamma_qpoint)

        # Use 2x2x2 supercell
        supercell_matrix = np.eye(3, dtype=int) * 2

        # Generate a displacement
        source_displacement = modes.generate_mode_displacement(
            q_index=0, mode_index=6, supercell_matrix=supercell_matrix, amplitude=1.0
        )

        # Create source supercell
        source_supercell_result = modes.generate_displaced_supercell(
            q_index=0,
            mode_index=6,
            supercell_matrix=supercell_matrix,
            amplitude=0.0,
            return_displacements=False,
        )
        source_supercell = _extract_atoms(source_supercell_result)

        # Create target supercell with both translation and shuffling
        np.random.seed(42)  # For reproducibility
        n_atoms = len(source_supercell)
        shuffle_indices = np.arange(n_atoms)
        np.random.shuffle(shuffle_indices)

        # Create shuffled and translated supercell
        target_supercell = copy.deepcopy(source_supercell)
        target_supercell.positions = target_supercell.positions[shuffle_indices]
        target_supercell.numbers = target_supercell.numbers[shuffle_indices]

        # Apply translation
        translation_vector = np.array([0.4, 0.6, 0.1])
        target_supercell.positions = target_supercell.positions + translation_vector

        # Apply periodic boundary conditions
        target_supercell.wrap()

        # Create correspondingly shuffled displacement
        target_displacement = source_displacement[shuffle_indices]

        # Project displacement
        coeff = project_displacements_between_supercells(
            source_displacement=source_displacement,
            target_displacement=target_displacement,
            source_supercell=source_supercell,
            target_supercell=target_supercell,
            normalize=True,
        )

        # Should be close to 1.0 (allowing for small errors from complex transformation)
        assert abs(coeff - 1.0) < 1e-5, (
            f"Combined translation and shuffling projection failed: coefficient = {coeff}"
        )

    def test_orthogonal_displacements(self):
        """
        Test projection between orthogonal displacements.
        Should return coefficient close to 0.0.
        """
        # Load BaTiO3 data
        gamma_qpoint = np.array([[0.0, 0.0, 0.0]])
        modes = PhononModes.from_phonopy_yaml(str(BATIO3_YAML_PATH), gamma_qpoint)

        # Use 2x2x2 supercell
        supercell_matrix = np.eye(3, dtype=int) * 2

        # Generate two different mode displacements (should be orthogonal)
        displacement1 = modes.generate_mode_displacement(
            q_index=0, mode_index=6, supercell_matrix=supercell_matrix, amplitude=1.0
        )
        displacement2 = modes.generate_mode_displacement(
            q_index=0, mode_index=7, supercell_matrix=supercell_matrix, amplitude=1.0
        )

        # Create identical supercells
        source_supercell_result = modes.generate_displaced_supercell(
            q_index=0,
            mode_index=6,
            supercell_matrix=supercell_matrix,
            amplitude=0.0,
            return_displacements=False,
        )
        source_supercell = _extract_atoms(source_supercell_result)
        target_supercell = copy.deepcopy(source_supercell)

        # Project orthogonal displacements
        coeff = project_displacements_between_supercells(
            source_displacement=displacement1,
            target_displacement=displacement2,
            source_supercell=source_supercell,
            target_supercell=target_supercell,
            normalize=True,
        )

        # Should be close to 0.0
        assert abs(coeff) < 1e-10, (
            f"Orthogonal displacements projection failed: coefficient = {coeff}"
        )

    def test_convenience_method(self):
        """
        Test the convenience method in PhononModes class.
        """
        # Load BaTiO3 data
        gamma_qpoint = np.array([[0.0, 0.0, 0.0]])
        modes = PhononModes.from_phonopy_yaml(str(BATIO3_YAML_PATH), gamma_qpoint)

        # Use 2x2x2 supercell
        supercell_matrix = np.eye(3, dtype=int) * 2

        # Generate a displacement
        displacement = modes.generate_mode_displacement(
            q_index=0, mode_index=6, supercell_matrix=supercell_matrix, amplitude=1.0
        )

        # Create supercells
        source_supercell_result = modes.generate_displaced_supercell(
            q_index=0,
            mode_index=6,
            supercell_matrix=supercell_matrix,
            amplitude=0.0,
            return_displacements=False,
        )
        source_supercell = _extract_atoms(source_supercell_result)
        target_supercell = copy.deepcopy(source_supercell)

        # Test convenience method
        coeff = modes.project_displacement_to_supercell(
            source_displacement=displacement,
            source_supercell=source_supercell,
            target_supercell=target_supercell,
            normalize=True,
        )

        # Should be 1.0 (identical displacement projected onto itself)
        assert abs(coeff - 1.0) < 1e-12, (
            f"Convenience method projection failed: coefficient = {coeff}"
        )

    def test_error_handling(self):
        """
        Test error handling for mismatched dimensions and invalid inputs.
        """
        # Load BaTiO3 data
        gamma_qpoint = np.array([[0.0, 0.0, 0.0]])
        modes = PhononModes.from_phonopy_yaml(str(BATIO3_YAML_PATH), gamma_qpoint)

        # Use 2x2x2 supercell
        supercell_matrix = np.eye(3, dtype=int) * 2

        # Create valid displacement and supercell
        displacement = modes.generate_mode_displacement(
            q_index=0, mode_index=6, supercell_matrix=supercell_matrix, amplitude=1.0
        )
        supercell_result = modes.generate_displaced_supercell(
            q_index=0,
            mode_index=6,
            supercell_matrix=supercell_matrix,
            amplitude=0.0,
            return_displacements=False,
        )
        supercell = _extract_atoms(supercell_result)

        # Test with wrong displacement dimensions
        wrong_displacement = np.random.rand(5, 3)  # Wrong number of atoms

        with pytest.raises(
            ValueError,
            match="Source displacement must match source supercell atom count",
        ):
            project_displacements_between_supercells(
                source_displacement=wrong_displacement,
                target_displacement=displacement,
                source_supercell=supercell,
                target_supercell=supercell,
            )

        with pytest.raises(
            ValueError,
            match="Target displacement must match target supercell atom count",
        ):
            project_displacements_between_supercells(
                source_displacement=displacement,
                target_displacement=wrong_displacement,
                source_supercell=supercell,
                target_supercell=supercell,
            )
