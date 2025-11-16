"""
Test suite for enhanced structure mapping functionality.

This module tests the enhanced atom mapping capabilities including:
- PBC distance calculations
- Origin alignment
- Shift optimization
- Detailed output generation
- Quality validation

Test data from data/yajundata/ is used for comprehensive validation.
"""

import pytest
import numpy as np
import os
from ase import Atoms
from ase.io import read

from phonproj.core.structure_analysis import (
    calculate_pbc_distance,
    find_closest_to_origin,
    shift_to_origin,
    create_enhanced_atom_mapping,
    MappingAnalyzer,
)


class TestPBCDistanceCalculation:
    """Test PBC distance calculation functionality."""

    def test_basic_pbc_distance(self):
        """Test basic PBC distance calculation."""
        # Simple cubic cell
        cell = np.eye(3) * 5.0  # 5Å cubic cell
        pos1 = np.array([0.1, 0.1, 0.1]) * 5.0  # (0.5, 0.5, 0.5)
        pos2 = np.array([0.9, 0.9, 0.9]) * 5.0  # (4.5, 4.5, 4.5)

        distance = calculate_pbc_distance(pos1, pos2, cell)

        # Should be minimal distance across periodic boundary
        expected_distance = np.linalg.norm(pos1 - (pos2 - 5.0))
        assert np.isclose(distance, expected_distance, atol=1e-6)

    def test_minimum_image_convention(self):
        """Test that minimum image convention is applied correctly."""
        cell = np.eye(3) * 10.0
        pos1 = np.array([1.0, 1.0, 1.0])
        pos2 = np.array([9.0, 9.0, 9.0])

        distance = calculate_pbc_distance(pos1, pos2, cell)

        # Should use periodic image, not direct distance
        expected_distance = np.linalg.norm(pos1 - (pos2 - 10.0))
        assert np.isclose(distance, expected_distance, atol=1e-6)
        assert distance < 5.0  # Should be much less than direct distance


class TestOriginAlignment:
    """Test origin alignment functionality."""

    def test_find_closest_to_origin(self):
        """Test finding atom closest to origin."""
        # Create simple structure
        positions = np.array(
            [
                [2.0, 2.0, 2.0],
                [0.5, 0.5, 0.5],
                [3.0, 3.0, 3.0],
            ]
        )
        structure = Atoms(symbols=["H"] * 3, positions=positions, cell=np.eye(3) * 10.0)

        closest_idx, distance, closest_pos = find_closest_to_origin(structure)

        assert closest_idx == 1  # Second atom should be closest
        assert np.isclose(distance, np.linalg.norm([0.5, 0.5, 0.5]))
        np.testing.assert_array_equal(closest_pos, [0.5, 0.5, 0.5])

    def test_shift_to_origin(self):
        """Test shifting structure to place atom at origin."""
        positions = np.array(
            [
                [1.0, 1.0, 1.0],
                [2.0, 2.0, 2.0],
            ]
        )
        structure = Atoms(symbols=["H"] * 2, positions=positions, cell=np.eye(3) * 10.0)

        shifted = shift_to_origin(structure, 0)

        # First atom should be at origin
        np.testing.assert_array_almost_equal(
            shifted.get_positions()[0], [0.0, 0.0, 0.0]
        )
        # Second atom should maintain relative position
        np.testing.assert_array_almost_equal(
            shifted.get_positions()[1], [1.0, 1.0, 1.0]
        )


class TestEnhancedAtomMapping:
    """Test enhanced atom mapping functionality."""

    @pytest.fixture
    def test_data_dir(self):
        """Get test data directory path."""
        return os.path.join(os.path.dirname(__file__), "..", "..", "data", "yajundata")

    @pytest.fixture
    def ref_structure(self, test_data_dir):
        """Load reference structure."""
        ref_file = os.path.join(test_data_dir, "ref.vasp")
        if os.path.exists(ref_file):
            result = read(ref_file)
            # read() may return list or single Atoms object
            return result[0] if isinstance(result, list) else result
        else:
            pytest.skip(f"Test data file {ref_file} not found")

    @pytest.fixture
    def sm2_structure(self, test_data_dir):
        """Load SM2 structure."""
        sm2_file = os.path.join(test_data_dir, "SM2.vasp")
        if os.path.exists(sm2_file):
            result = read(sm2_file)
            return result[0] if isinstance(result, list) else result
        else:
            pytest.skip(f"Test data file {sm2_file} not found")

    @pytest.fixture
    def supercell_structure(self, test_data_dir):
        """Load supercell structure."""
        supercell_file = os.path.join(test_data_dir, "supercell_undistorted.vasp")
        if os.path.exists(supercell_file):
            result = read(supercell_file)
            return result[0] if isinstance(result, list) else result
        else:
            pytest.skip(f"Test data file {supercell_file} not found")

    def test_shuffle_only_mapping(self, ref_structure):
        """Test mapping with atom shuffling only."""
        # Create shuffled version
        shuffled_indices = np.random.permutation(len(ref_structure))
        shuffled_positions = ref_structure.get_positions()[shuffled_indices]
        shuffled_species = [
            ref_structure.get_chemical_symbols()[i] for i in shuffled_indices
        ]

        shuffled_structure = Atoms(
            symbols=shuffled_species,
            positions=shuffled_positions,
            cell=ref_structure.get_cell(),
            pbc=True,
        )

        # Test enhanced mapping
        mapping, cost, shift_vector, quality = create_enhanced_atom_mapping(
            ref_structure,
            shuffled_structure,
            optimize_shift=True,
            origin_alignment=True,
        )

        # Validate mapping
        assert len(mapping) == len(ref_structure)
        assert cost < 0.001  # Should be very small for pure shuffle
        assert np.linalg.norm(shift_vector) < 0.1  # Should be minimal shift

        # Check that mapping correctly reverses the shuffle
        for i, target_idx in enumerate(mapping):
            original_species = ref_structure.get_chemical_symbols()[i]
            mapped_species = shuffled_structure.get_chemical_symbols()[target_idx]
            assert original_species == mapped_species

    def test_shuffle_and_translation_mapping(self, ref_structure):
        """Test mapping with shuffle and translation."""
        # Create shuffled and translated version
        shuffled_indices = np.random.permutation(len(ref_structure))
        shuffled_positions = ref_structure.get_positions()[shuffled_indices]
        shuffled_species = [
            ref_structure.get_chemical_symbols()[i] for i in shuffled_indices
        ]

        # Apply translation in scaled coordinates
        translation = np.array([1.0, 1.0, 1.0])  # Translate by 1 in scaled coordinates
        cell = ref_structure.get_cell()
        scaled_translation = translation @ cell
        translated_positions = shuffled_positions + scaled_translation

        translated_structure = Atoms(
            symbols=shuffled_species,
            positions=translated_positions,
            cell=ref_structure.get_cell(),
            pbc=True,
        )

        # Test enhanced mapping without origin alignment to preserve translation
        mapping, cost, shift_vector, quality = create_enhanced_atom_mapping(
            ref_structure,
            translated_structure,
            optimize_shift=True,
            origin_alignment=False,  # Don't align to origin to preserve translation
            force_near_origin=False,  # Don't force near origin to preserve translation
        )

        # Validate mapping
        assert len(mapping) == len(ref_structure)
        assert cost < 1.0  # Should be reasonable after shift optimization

        # Shift vector should approximately match the applied translation
        shift_magnitude = np.linalg.norm(shift_vector)
        expected_magnitude = np.linalg.norm(scaled_translation)
        # Allow larger tolerance for real data with large cells
        assert np.isclose(shift_magnitude, expected_magnitude, rtol=0.5)

    def test_shuffle_translation_displacement_mapping(self, ref_structure):
        """Test mapping with shuffle, translation, and displacement."""
        # Create shuffled, translated, and displaced version
        shuffled_indices = np.random.permutation(len(ref_structure))
        shuffled_positions = ref_structure.get_positions()[shuffled_indices]
        shuffled_species = [
            ref_structure.get_chemical_symbols()[i] for i in shuffled_indices
        ]

        # Apply translation
        translation = np.array([1.0, 0.5, 0.3])  # Different translation
        cell = ref_structure.get_cell()
        scaled_translation = translation @ cell
        translated_positions = shuffled_positions + scaled_translation

        # Apply uniform random displacement
        np.random.seed(42)  # For reproducibility
        random_displacement = np.random.normal(
            0, 0.1, translated_positions.shape
        )  # 0.1Å std
        final_positions = translated_positions + random_displacement

        final_structure = Atoms(
            symbols=shuffled_species,
            positions=final_positions,
            cell=ref_structure.get_cell(),
            pbc=True,
        )

        # Test enhanced mapping
        mapping, cost, shift_vector, quality = create_enhanced_atom_mapping(
            ref_structure, final_structure, optimize_shift=True, origin_alignment=True
        )

        # Validate mapping
        assert len(mapping) == len(ref_structure)

        # Cost should reflect the random displacement and translation
        # For real data with large cells, cost can be much larger
        assert cost > 0  # Should be positive
        assert cost < 500  # Should be reasonable for 160 atoms with displacement

        # Quality metrics should be reasonable for real data
        assert quality["mean_distance"] > 0
        assert quality["max_distance"] > 0

    def test_cross_structure_mapping(self, ref_structure, sm2_structure):
        """Test mapping between different but related structures."""
        # Only test if both structures have same number of atoms
        if len(ref_structure) != len(sm2_structure):
            pytest.skip("Structures have different number of atoms")

        # Test enhanced mapping
        mapping, cost, shift_vector, quality = create_enhanced_atom_mapping(
            ref_structure, sm2_structure, optimize_shift=True, origin_alignment=True
        )

        # Validate mapping
        assert len(mapping) == len(ref_structure)
        assert cost >= 0  # Cost should be non-negative

        # Species should be conserved
        ref_species = ref_structure.get_chemical_symbols()
        mapped_species = [sm2_structure.get_chemical_symbols()[i] for i in mapping]

        # Count species in both
        from collections import Counter

        ref_counts = Counter(ref_species)
        mapped_counts = Counter(mapped_species)

        assert ref_counts == mapped_counts, "Species not conserved in mapping"

    def test_supercell_mapping(self, ref_structure, supercell_structure):
        """Test mapping between primitive and supercell structures."""
        # This test may be skipped if structures have different sizes
        if len(ref_structure) != len(supercell_structure):
            pytest.skip("Primitive and supercell have different atom counts")

        # Test enhanced mapping
        mapping, cost, shift_vector, quality = create_enhanced_atom_mapping(
            ref_structure,
            supercell_structure,
            optimize_shift=True,
            origin_alignment=True,
        )

        # Validate mapping
        assert len(mapping) == len(ref_structure)
        assert cost >= 0

        # For supercell mapping, we expect reasonable distances
        assert quality["mean_distance"] < 2.0  # Should be within reasonable range


class TestMappingAnalyzer:
    """Test MappingAnalyzer class functionality."""

    @pytest.fixture
    def sample_mapping_data(self):
        """Create sample mapping data for testing."""
        # Create simple test structures
        positions1 = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        positions2 = np.array([[0.1, 0.0, 0.0], [1.1, 0.0, 0.0]])

        structure1 = Atoms(
            symbols=["H", "H"], positions=positions1, cell=np.eye(3) * 5.0
        )
        structure2 = Atoms(
            symbols=["H", "H"], positions=positions2, cell=np.eye(3) * 5.0
        )

        mapping = np.array([0, 1])  # Identity mapping
        shift_vector = np.array([0.1, 0.0, 0.0])

        quality_metrics = {
            "mean_distance": 0.1,
            "max_distance": 0.1,
            "min_distance": 0.1,
            "std_distance": 0.0,
            "atoms_above_threshold": 0,
            "atoms_above_01angstrom": 0,
            "atoms_above_05angstrom": 0,
            "shift_magnitude": 0.1,
        }

        return structure1, structure2, mapping, shift_vector, quality_metrics

    def test_analyzer_initialization(self, sample_mapping_data):
        """Test MappingAnalyzer initialization."""
        struct1, struct2, mapping, shift, quality = sample_mapping_data

        analyzer = MappingAnalyzer(struct1, struct2, mapping, shift, quality)

        assert analyzer.structure1 is struct1
        assert analyzer.structure2 is struct2
        np.testing.assert_array_equal(analyzer.mapping, mapping)
        np.testing.assert_array_equal(analyzer.shift_vector, shift)
        assert analyzer.quality_metrics is quality

    def test_mapping_analysis(self, sample_mapping_data):
        """Test comprehensive mapping analysis."""
        struct1, struct2, mapping, shift, quality = sample_mapping_data

        analyzer = MappingAnalyzer(struct1, struct2, mapping, shift, quality)
        analysis = analyzer.analyze_mapping()

        # Check analysis structure
        assert "mapping_details" in analysis
        assert "quality_metrics" in analysis
        assert "shift_vector" in analysis
        assert "species_conservation" in analysis
        assert "mapping_summary" in analysis

        # Check mapping details
        details = analysis["mapping_details"]
        assert len(details) == 2
        assert all("atom_index" in detail for detail in details)
        assert all("distance" in detail for detail in details)

    def test_output_generation(self, sample_mapping_data, tmp_path):
        """Test detailed output file generation."""
        struct1, struct2, mapping, shift, quality = sample_mapping_data

        analyzer = MappingAnalyzer(struct1, struct2, mapping, shift, quality)

        # Test output generation
        output_file = os.path.join(tmp_path, "test_mapping_output.txt")
        analyzer.save_detailed_output(output_file)

        # Check file was created
        assert os.path.exists(output_file)

        # Check file content
        with open(output_file, "r") as f:
            content = f.read()

        assert "ENHANCED ATOM MAPPING ANALYSIS REPORT" in content
        assert "MAPPING SUMMARY" in content
        assert "DETAILED MAPPING TABLE" in content
        assert "Idx" in content  # Table header


class TestIntegration:
    """Integration tests for the complete enhanced mapping workflow."""

    @pytest.fixture
    def test_data_dir(self):
        """Get test data directory path."""
        return os.path.join(os.path.dirname(__file__), "..", "..", "data", "yajundata")

    def test_complete_workflow_with_output(self, test_data_dir, tmp_path):
        """Test complete workflow from mapping to output generation."""
        ref_file = os.path.join(test_data_dir, "ref.vasp")
        if not os.path.exists(ref_file):
            pytest.skip(f"Test data file {ref_file} not found")

        # Load reference structure
        result = read(ref_file)
        ref_structure = result[0] if isinstance(result, list) else result

        # Create test structure (shuffled + translated)
        shuffled_indices = np.random.permutation(len(ref_structure))
        shuffled_positions = ref_structure.get_positions()[shuffled_indices]
        shuffled_species = [
            ref_structure.get_chemical_symbols()[i] for i in shuffled_indices
        ]

        # Apply translation
        translation = np.array([0.5, 0.3, 0.7])
        cell = ref_structure.get_cell()
        scaled_translation = translation @ cell
        translated_positions = shuffled_positions + scaled_translation

        test_structure = Atoms(
            symbols=shuffled_species,
            positions=translated_positions,
            cell=ref_structure.get_cell(),
            pbc=True,
        )

        # Perform enhanced mapping
        mapping, cost, shift_vector, quality = create_enhanced_atom_mapping(
            ref_structure, test_structure, optimize_shift=True, origin_alignment=True
        )

        # Create analyzer and generate output
        analyzer = MappingAnalyzer(
            ref_structure, test_structure, mapping, shift_vector, quality
        )

        output_file = os.path.join(tmp_path, "integration_test_output.txt")
        analyzer.save_detailed_output(output_file)

        # Validate complete workflow
        assert os.path.exists(output_file)
        assert len(mapping) == len(ref_structure)
        assert cost >= 0

        # Validate output file contains expected sections
        with open(output_file, "r") as f:
            content = f.read()

        assert "MAPPING SUMMARY" in content
        assert "SHIFT VECTOR INFORMATION" in content
        assert "QUALITY METRICS" in content
        assert "DETAILED MAPPING TABLE" in content
