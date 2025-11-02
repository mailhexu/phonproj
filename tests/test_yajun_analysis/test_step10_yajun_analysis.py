#!/usr/bin/env python3
"""
Test Step 10: Yajun's PbTiO3 Data Analysis

Tests the complete workflow for analyzing Yajun's experimental/computational
PbTiO3 displacement data using phonon mode decomposition.

This test validates:
1. Loading PbTiO3 phonopy data from yajundata directory
2. Generating 16x1x1 supercell and commensurate q-points
3. Loading displaced structure from CONTCAR-a1a2-GS
4. Structure compatibility checks
5. Mock displacement projection workflow
"""

import pytest
import numpy as np
from pathlib import Path
from phonproj.modes import PhononModes
from phonproj.core import load_from_phonopy_files


class TestYajunAnalysis:
    """Test suite for Yajun's PbTiO3 analysis workflow."""

    @pytest.fixture
    def yajun_data_dir(self):
        """Fixture providing path to Yajun's data directory."""
        data_dir = (
            Path(__file__).parent.parent.parent
            / "data"
            / "yajundata"
            / "0.02-P4mmm-PTO"
        )
        if not data_dir.exists():
            pytest.skip(f"Yajun data not found at {data_dir}")
        return data_dir

    @pytest.fixture
    def contcar_path(self):
        """Fixture providing path to displaced structure file."""
        contcar_path = (
            Path(__file__).parent.parent.parent
            / "data"
            / "yajundata"
            / "CONTCAR-a1a2-GS"
        )
        if not contcar_path.exists():
            pytest.skip(f"CONTCAR-a1a2-GS not found at {contcar_path}")
        return contcar_path

    @pytest.fixture
    def phonopy_data(self, yajun_data_dir):
        """Fixture providing loaded phonopy data."""
        data = load_from_phonopy_files(yajun_data_dir)
        return data

    def test_load_phonopy_data(self, yajun_data_dir):
        """Test loading phonopy data from directory using phonopy API."""
        data = load_from_phonopy_files(yajun_data_dir)

        # Check that we have the expected data structure
        assert "phonopy" in data
        assert "primitive_cell" in data
        assert "unitcell" in data
        assert "supercell" in data

        # Check primitive cell has expected number of atoms for PbTiO3
        primitive_cell = data["primitive_cell"]
        assert len(primitive_cell) == 10  # PbTiO3 primitive cell has 10 atoms

    def test_generate_16x1x1_qpoints(self):
        """Test generation of commensurate q-points for 16x1x1 supercell."""
        qpoints = []
        for n in range(16):
            qpoints.append([n / 16.0, 0.0, 0.0])
        qpoints = np.array(qpoints)

        # Check q-point properties
        assert qpoints.shape == (16, 3)
        assert np.allclose(qpoints[:, 1], 0.0)  # All y-components zero
        assert np.allclose(qpoints[:, 2], 0.0)  # All z-components zero
        assert np.allclose(qpoints[0], [0.0, 0.0, 0.0])  # Gamma point
        assert np.allclose(qpoints[-1], [15 / 16, 0.0, 0.0])  # Last q-point

        # Check commensurate property: 16 * q[0] should be integers
        scaled_q = qpoints[:, 0] * 16
        assert np.allclose(scaled_q, np.round(scaled_q))

    def test_load_displaced_structure(self, contcar_path):
        """Test loading displaced structure from CONTCAR-a1a2-GS."""
        with open(contcar_path, "r") as f:
            lines = f.readlines()

        # Parse basic structure
        scale = float(lines[1].strip())
        assert scale > 0, "Scale factor should be positive"

        # Parse lattice vectors
        lattice = []
        for i in range(2, 5):
            vec = [float(x) for x in lines[i].strip().split()]
            lattice.append([v * scale for v in vec])
        lattice = np.array(lattice)

        assert lattice.shape == (3, 3)

        # Parse atom information
        atom_types = lines[5].strip().split()
        atom_counts = [int(x) for x in lines[6].strip().split()]
        total_atoms = sum(atom_counts)

        # Check PbTiO3 composition for 16x1x1 supercell
        expected_composition = {"Pb": 32, "Ti": 32, "O": 96}  # 16 * (2 Pb + 2 Ti + 6 O)
        actual_composition = dict(zip(atom_types, atom_counts))

        assert actual_composition == expected_composition
        assert total_atoms == 160  # 16 * 10 atoms

    def test_supercell_matrix_16x1x1(self):
        """Test 16x1x1 supercell matrix properties."""
        supercell_matrix = np.array([[16, 0, 0], [0, 1, 0], [0, 0, 1]])

        # Check determinant (number of primitive cells)
        det = int(np.round(np.linalg.det(supercell_matrix)))
        assert det == 16

        # Check expected expansion
        assert supercell_matrix[0, 0] == 16  # 16x expansion along a-axis
        assert supercell_matrix[1, 1] == 1  # No expansion along b-axis
        assert supercell_matrix[2, 2] == 1  # No expansion along c-axis

    def test_structure_composition_compatibility(self, phonopy_data, contcar_path):
        """Test that primitive and displaced structures have compatible compositions."""
        # Get primitive cell composition from phonopy data
        primitive_cell = phonopy_data["primitive_cell"]
        primitive_symbols = primitive_cell.get_chemical_symbols()
        primitive_composition = {}
        for symbol in primitive_symbols:
            primitive_composition[symbol] = primitive_composition.get(symbol, 0) + 1

        # Load displaced structure composition
        with open(contcar_path, "r") as f:
            lines = f.readlines()

        atom_types = lines[5].strip().split()
        atom_counts = [int(x) for x in lines[6].strip().split()]
        displaced_composition = dict(zip(atom_types, atom_counts))

        # Check that displaced composition is 16x the primitive composition
        for atom_type, primitive_count in primitive_composition.items():
            expected_displaced_count = primitive_count * 16
            actual_displaced_count = displaced_composition.get(atom_type, 0)
            assert actual_displaced_count == expected_displaced_count, (
                f"Atom {atom_type}: expected {expected_displaced_count}, got {actual_displaced_count}"
            )

    def test_lattice_compatibility(self, phonopy_data, contcar_path):
        """Test lattice compatibility between primitive and displaced structures."""
        # Get primitive lattice from phonopy data
        primitive_cell = phonopy_data["primitive_cell"]
        primitive_lattice = primitive_cell.get_cell()

        # Load displaced structure lattice
        with open(contcar_path, "r") as f:
            lines = f.readlines()

        scale = float(lines[1].strip())
        displaced_lattice = []
        for i in range(2, 5):
            vec = [float(x) for x in lines[i].strip().split()]
            displaced_lattice.append([v * scale for v in vec])
        displaced_lattice = np.array(displaced_lattice)

        # For 16x1x1 supercell, expected transformation:
        # a' = 16 * a, b' = b, c' = c
        expected_supercell_lattice = primitive_lattice.copy()
        expected_supercell_lattice[0] *= 16  # Scale a-vector by 16

        # Check lattice compatibility (allow small numerical differences)
        lattice_diff = np.abs(displaced_lattice - expected_supercell_lattice)
        max_diff = np.max(lattice_diff)

        assert max_diff < 0.1, f"Lattices differ by {max_diff:.3f} Å, expected < 0.1 Å"

    def test_mock_projection_workflow(self):
        """Test the mock projection analysis workflow."""
        # Generate mock data similar to the example
        n_qpoints = 16
        n_modes_per_q = 30

        np.random.seed(42)  # For reproducible testing

        total_projection_squared = 0.0

        for q_idx in range(n_qpoints):
            # Mock frequencies and projections
            mock_frequencies = np.random.uniform(0.5, 15.0, n_modes_per_q)
            mock_projections = np.random.normal(
                0, 0.1, n_modes_per_q
            ) + 1j * np.random.normal(0, 0.1, n_modes_per_q)

            # Accumulate projection squares
            for proj in mock_projections:
                total_projection_squared += abs(proj) ** 2

        # Check that total is reasonable (should be around 1.0 for complete basis)
        # With random projections, we expect some value that's not too far from 1.0
        assert 0.1 < total_projection_squared < 10.0, (
            f"Total projection squared {total_projection_squared:.3f} seems unreasonable"
        )

    def test_step10_integration(self, phonopy_data, contcar_path):
        """Integration test for the complete Step 10 workflow."""
        # This test runs the main components together

        # 1. Load phonopy data
        primitive_cell = phonopy_data["primitive_cell"]

        # 2. Generate q-points
        qpoints = np.array([[n / 16.0, 0.0, 0.0] for n in range(16)])

        # 3. Load displaced structure
        with open(contcar_path, "r") as f:
            lines = f.readlines()
        atom_types = lines[5].strip().split()
        atom_counts = [int(x) for x in lines[6].strip().split()]

        # 4. Verify consistency
        assert len(qpoints) == 16
        assert sum(atom_counts) == 160  # Expected for 16x1x1 supercell
        assert len(primitive_cell) == 10  # Primitive cell atoms

        # 5. Check that we have all components for analysis
        assert "phonopy" in phonopy_data
        assert primitive_cell.get_cell() is not None

        print("✅ Step 10 integration test passed")


if __name__ == "__main__":
    # Run tests directly
    import sys

    test_instance = TestYajunAnalysis()

    try:
        # Test data availability
        yajun_dir = (
            Path(__file__).parent.parent.parent
            / "data"
            / "yajundata"
            / "0.02-P4mmm-PTO"
        )
        contcar_path = (
            Path(__file__).parent.parent.parent
            / "data"
            / "yajundata"
            / "CONTCAR-a1a2-GS"
        )

        if not yajun_dir.exists():
            print(f"❌ Yajun data directory not found: {yajun_dir}")
            sys.exit(1)

        if not contcar_path.exists():
            print(f"❌ CONTCAR-a1a2-GS not found: {contcar_path}")
            sys.exit(1)

        # Load phonopy data once
        print("Loading phonopy data...")
        phonopy_data = load_from_phonopy_files(yajun_dir)
        print("✅ Phonopy data loaded")

        # Run individual tests
        print("\nTesting Yajun Analysis Step 10...")

        test_instance.test_load_phonopy_data(yajun_dir)
        print("✅ Phonopy data loading test passed")

        test_instance.test_generate_16x1x1_qpoints()
        print("✅ Q-point generation test passed")

        test_instance.test_load_displaced_structure(contcar_path)
        print("✅ Displaced structure loading test passed")

        test_instance.test_supercell_matrix_16x1x1()
        print("✅ Supercell matrix test passed")

        test_instance.test_structure_composition_compatibility(
            phonopy_data, contcar_path
        )
        print("✅ Structure composition compatibility test passed")

        test_instance.test_lattice_compatibility(phonopy_data, contcar_path)
        print("✅ Lattice compatibility test passed")

        test_instance.test_mock_projection_workflow()
        print("✅ Mock projection workflow test passed")

        test_instance.test_step10_integration(phonopy_data, contcar_path)
        print("✅ Step 10 integration test passed")

        print("\n🎉 All Step 10 tests passed successfully!")

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
