"""
Tests for the Phonon Displacement Generator
"""

import pytest
import numpy as np
from pathlib import Path
import tempfile
import os

from phonproj.displacement import PhononDisplacementGenerator


class TestPhononDisplacementGenerator:
    """Test the PhononDisplacementGenerator class."""

    @pytest.fixture
    def batio3_data(self):
        """Get BaTiO3 phonopy data path."""
        return (
            Path(__file__).parent.parent.parent / "data" / "BaTiO3_phonopy_params.yaml"
        )

    @pytest.fixture
    def generator(self, batio3_data):
        """Create a displacement generator instance."""
        return PhononDisplacementGenerator(str(batio3_data))

    @pytest.fixture
    def supercell_2x2x2(self):
        """2x2x2 supercell matrix."""
        return np.diag([2, 2, 2])

    def test_initialization(self, batio3_data):
        """Test generator initialization."""
        gen = PhononDisplacementGenerator(str(batio3_data))
        assert gen.phonopy_path == batio3_data
        assert gen.phonopy_data is not None
        assert gen.phonon_modes is None

    def test_calculate_modes(self, generator, supercell_2x2x2):
        """Test mode calculation."""
        modes = generator.calculate_modes(supercell_2x2x2)

        assert modes is not None
        assert generator.phonon_modes is not None
        assert len(modes.qpoints) == 8  # 2x2x2 = 8 q-points
        assert modes.frequencies.shape == (8, 15)  # 8 q-points, 15 modes each
        assert modes.n_modes == 15

    def test_get_commensurate_qpoints(self, generator, supercell_2x2x2):
        """Test getting commensurate q-points."""
        qpoints = generator.get_commensurate_qpoints(supercell_2x2x2)

        assert isinstance(qpoints, list)
        assert len(qpoints) == 8  # All 8 q-points should be commensurate
        assert all(isinstance(q, int) for q in qpoints)

    def test_generate_displacement(self, generator, supercell_2x2x2):
        """Test displacement generation for a specific mode."""
        # Test with Gamma point, first optical mode
        displacement = generator.generate_displacement(
            q_idx=0, mode_idx=3, supercell_matrix=supercell_2x2x2, amplitude=0.1
        )

        assert isinstance(displacement, np.ndarray)
        assert displacement.shape[1] == 3  # 3D displacements
        assert len(displacement) > 0  # Should have atoms

    def test_generate_supercell_structure(self, generator, supercell_2x2x2):
        """Test supercell structure generation."""
        structure = generator.generate_supercell_structure(
            q_idx=0, mode_idx=3, supercell_matrix=supercell_2x2x2, amplitude=0.1
        )

        from ase import Atoms

        assert isinstance(structure, Atoms)
        assert len(structure) > 0  # Should have atoms
        assert structure.get_positions().shape[1] == 3  # 3D positions

    def test_print_displacements(self, generator, supercell_2x2x2, capsys):
        """Test displacement printing (capture output)."""
        generator.print_displacements(
            supercell_matrix=supercell_2x2x2, amplitude=0.05, max_atoms_per_mode=2
        )

        captured = capsys.readouterr()
        assert "Supercell Displacements" in captured.out
        assert "Found 8 commensurate q-points" in captured.out
        assert "Q-point 0:" in captured.out
        assert "Mode" in captured.out
        assert "freq" in captured.out

    def test_save_all_structures(self, generator, supercell_2x2x2):
        """Test saving all structures to directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            result = generator.save_all_structures(
                supercell_matrix=supercell_2x2x2, output_dir=temp_dir, amplitude=0.05
            )

            # Check result summary
            assert result["total_saved"] == 120  # 8 q-points × 15 modes
            assert result["amplitude"] == 0.05
            assert result["output_dir"] == temp_dir
            assert len(result["saved_files"]) == 120

            # Check that files were actually created
            output_path = Path(temp_dir)
            vasp_files = list(output_path.glob("*.vasp"))
            assert len(vasp_files) == 120

            # Check filename format
            sample_file = vasp_files[0]
            assert "q" in sample_file.name
            assert "mode" in sample_file.name
            assert "freq" in sample_file.name
            assert sample_file.suffix == ".vasp"

    def test_different_supercell_sizes(self, generator):
        """Test with different supercell sizes."""
        # Test 4x1x1 supercell
        supercell_4x1x1 = np.diag([4, 1, 1])
        qpoints = generator.get_commensurate_qpoints(supercell_4x1x1)
        assert len(qpoints) == 4  # 4x1x1 = 4 q-points

        # Test 3x3x3 supercell
        supercell_3x3x3 = np.diag([3, 3, 3])
        qpoints = generator.get_commensurate_qpoints(supercell_3x3x3)
        assert len(qpoints) == 27  # 3x3x3 = 27 q-points

    def test_amplitude_scaling(self, generator, supercell_2x2x2):
        """Test that amplitude properly scales displacements."""
        amp1 = 0.05
        amp2 = 0.1

        disp1 = generator.generate_displacement(
            q_idx=0, mode_idx=3, supercell_matrix=supercell_2x2x2, amplitude=amp1
        )
        disp2 = generator.generate_displacement(
            q_idx=0, mode_idx=3, supercell_matrix=supercell_2x2x2, amplitude=amp2
        )

        # Second amplitude should be 2x the first
        ratio = np.max(np.abs(disp2)) / np.max(np.abs(disp1))
        assert abs(ratio - 2.0) < 1e-10  # Allow for numerical precision

    def test_error_handling(self, generator):
        """Test error handling for invalid inputs."""
        # Test invalid supercell matrix
        with pytest.raises(Exception):
            generator.calculate_modes(np.array([[1, 2], [3, 4]]))  # Not 3x3

        # Test invalid q-point index
        supercell_2x2x2 = np.diag([2, 2, 2])
        generator.calculate_modes(supercell_2x2x2)

        with pytest.raises(Exception):
            generator.generate_displacement(
                q_idx=100, mode_idx=0, supercell_matrix=supercell_2x2x2
            )  # Invalid q-point index
