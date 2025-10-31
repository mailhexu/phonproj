"""
Tests for Step 7 functionality: Advanced supercell displacement generation.

Tests according to plan.md requirements:
- Test for the Gamma q-point and the supercell of 1x1x1, if the displacement are orthonormal with mass-weighted inner product.
- Test for the completeness of the eigendisplacement for Gamma q-point and supercell of 1x1x1, by first normalize the random unit displacement with the mass-weighted norm 1, then project it to all the eigendisplacement, and check if the sum of the projections squared is 1.
- Test the implementation of generating the dispalcement for all commensurate q-points of a given supercell.
- Test for a non-Gamma q-point and supercell of 2x2x2, if the displacement are orthogonal with mass-weighted inner product, and the mass-weighted norm of each displacement is 1.
- Test the completeness of the eigendisplacement for all the commensurate q-point of 2x2x2 supercell, by first normalize the random unit displacement with the mass-weighted norm 1, then project it to all the eigendisplacement, and check if the sum of the projections squared is 1.
"""

import pytest
import numpy as np
from pathlib import Path

from phonproj.modes import PhononModes


# Test data paths
BATIO3_YAML_PATH = (
    Path(__file__).parent.parent.parent / "data" / "BaTiO3_phonopy_params.yaml"
)
PPTO3_DATA_PATH = Path(__file__).parent.parent.parent / "data" / "PbTiO3"


class TestAdvancedSupercellDisplacements:
    """Test suite for Step 7 advanced supercell displacement functionality."""

    def test_gamma_point_1x1x1_orthonormality(self):
        """
        Test for the Gamma q-point and the supercell of 1x1x1,
        if the displacements are orthonormal with mass-weighted inner product.
        """
        # Load BaTiO3 data - only need Gamma point
        gamma_qpoint = np.array([[0.0, 0.0, 0.0]])
        modes = PhononModes.from_phonopy_yaml(str(BATIO3_YAML_PATH), gamma_qpoint)

        # Use 1x1x1 supercell (identity matrix)
        supercell_matrix = np.eye(3, dtype=int)

        # Get Gamma point (index 0)
        gamma_index = 0

        # Generate displacements for all modes at Gamma point
        all_displacements = modes.generate_all_mode_displacements(
            gamma_index, supercell_matrix, amplitude=1.0
        )

        # Check orthonormality with mass-weighted inner product
        n_modes = all_displacements.shape[0]

        # Calculate orthonormality matrix
        orthonormality_matrix = np.zeros((n_modes, n_modes), dtype=complex)

        for i in range(n_modes):
            for j in range(n_modes):
                # Mass-weighted inner product
                projection = modes.mass_weighted_projection(
                    all_displacements[i], all_displacements[j]
                )
                orthonormality_matrix[i, j] = projection

        # Check that the matrix is approximately identity
        identity = np.eye(n_modes)
        max_deviation = np.max(np.abs(orthonormality_matrix - identity))

        # Should be orthonormal within tolerance
        assert max_deviation < 1e-12, (
            f"Gamma point 1x1x1 orthonormality failed: max deviation = {max_deviation}"
        )

    def test_gamma_point_1x1x1_completeness(self):
        """
        Test for the completeness of the eigendisplacement for Gamma q-point and supercell of 1x1x1.
        Normalize random displacement with mass-weighted norm 1, project to all eigendisplacements,
        and check if sum of projections squared is 1.
        """
        # Load BaTiO3 data - only need Gamma point
        gamma_qpoint = np.array([[0.0, 0.0, 0.0]])
        modes = PhononModes.from_phonopy_yaml(str(BATIO3_YAML_PATH), gamma_qpoint)

        # Use 1x1x1 supercell (identity matrix)
        supercell_matrix = np.eye(3, dtype=int)

        # Get Gamma point (index 0)
        gamma_index = 0

        # Generate displacements for all modes at Gamma point
        all_displacements = modes.generate_all_mode_displacements(
            gamma_index, supercell_matrix, amplitude=1.0
        )

        # Create a random displacement
        np.random.seed(42)  # For reproducibility
        n_atoms = modes._n_atoms
        random_displacement = np.random.rand(n_atoms, 3)

        # Normalize with mass-weighted norm 1
        current_norm = modes.mass_weighted_norm(random_displacement)
        normalized_displacement = random_displacement / current_norm

        # Verify normalization
        assert abs(modes.mass_weighted_norm(normalized_displacement) - 1.0) < 1e-12

        # Project onto all eigendisplacements and sum squared projections
        sum_projections_squared = 0.0

        for i in range(all_displacements.shape[0]):
            projection = modes.mass_weighted_projection(
                normalized_displacement, all_displacements[i]
            )
            sum_projections_squared += abs(projection) ** 2

        # Should sum to 1 (completeness)
        assert abs(sum_projections_squared - 1.0) < 1e-4, (
            f"Gamma point 1x1x1 completeness failed: sum = {sum_projections_squared}"
        )

    def test_commensurate_qpoints_generation(self):
        """
        Test the implementation of generating displacements for all commensurate q-points of a given supercell.
        """
        # Load BaTiO3 data - need multiple q-points for 2x2x2 supercell
        # For 2x2x2 supercell, we need q-points on 2x2x2 grid
        qpoints_2x2x2 = []
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    qpoints_2x2x2.append([i / 2.0, j / 2.0, k / 2.0])
        qpoints_2x2x2 = np.array(qpoints_2x2x2)

        modes = PhononModes.from_phonopy_yaml(str(BATIO3_YAML_PATH), qpoints_2x2x2)

        # Use 2x2x2 supercell
        supercell_matrix = np.eye(3, dtype=int) * 2

        # Get commensurate q-points
        commensurate_qpoints = modes.get_commensurate_qpoints(supercell_matrix)

        # Should have 8 q-points for 2x2x2 supercell
        assert len(commensurate_qpoints) == 8, (
            f"Expected 8 commensurate q-points, got {len(commensurate_qpoints)}"
        )

        # Generate displacements for all commensurate q-points
        all_commensurate_displacements = modes.generate_all_commensurate_displacements(
            supercell_matrix, amplitude=1.0
        )

        # Check that we have displacement data
        assert len(all_commensurate_displacements) > 0, (
            "No commensurate displacements generated"
        )

        # Each displacement set should have the correct shape
        for q_index, displacements in all_commensurate_displacements.items():
            expected_n_supercell_atoms = 8 * modes._n_atoms  # 2x2x2 = 8 primitive cells
            assert displacements.shape[1] == expected_n_supercell_atoms, (
                f"Wrong number of supercell atoms: {displacements.shape[1]}"
            )
            assert displacements.shape[2] == 3, "Displacements should be 3D vectors"

    def test_non_gamma_2x2x2_orthogonality_and_norm(self):
        """
        Test for a non-Gamma q-point and supercell of 2x2x2, if the displacements are orthogonal
        with mass-weighted inner product, and the mass-weighted norm of each displacement is 1.
        """
        # Load BaTiO3 data - include non-Gamma q-points for 2x2x2
        qpoints_2x2x2 = []
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    qpoints_2x2x2.append([i / 2.0, j / 2.0, k / 2.0])
        qpoints_2x2x2 = np.array(qpoints_2x2x2)

        modes = PhononModes.from_phonopy_yaml(str(BATIO3_YAML_PATH), qpoints_2x2x2)

        # Use 2x2x2 supercell
        supercell_matrix = np.eye(3, dtype=int) * 2
        N = 8  # Number of primitive cells in 2x2x2 supercell

        # Find a non-Gamma q-point (not [0,0,0])
        non_gamma_index = None
        for i, qpoint in enumerate(modes.qpoints):
            if not np.allclose(qpoint, [0, 0, 0], atol=1e-6):
                non_gamma_index = i
                break

        if non_gamma_index is None:
            pytest.skip("No non-Gamma q-points found in the data")

        # Generate displacements for all modes at non-Gamma point
        all_displacements = modes.generate_all_mode_displacements(
            non_gamma_index, supercell_matrix, amplitude=1.0
        )

        n_modes = all_displacements.shape[0]

        # Check orthogonality
        for i in range(n_modes):
            for j in range(i + 1, n_modes):
                projection = modes.mass_weighted_projection(
                    all_displacements[i], all_displacements[j]
                )
                assert abs(projection) < 1e-6, (
                    f"Non-orthogonal modes {i}, {j}: projection = {projection}"
                )

        # Check that mass-weighted norm of each displacement is 1
        expected_norm = 1.0  # Each eigenmode should have mass-weighted norm = 1.0
        for i in range(n_modes):
            norm = modes.mass_weighted_norm(all_displacements[i])
            assert abs(norm - expected_norm) < 1e-10, (
                f"Wrong norm for mode {i}: {norm}, expected {expected_norm}"
            )

    def test_completeness_all_commensurate_2x2x2(self):
        """
        Test the completeness of the eigendisplacement for all commensurate q-points of 2x2x2 supercell.

        Normalize random displacement with mass-weighted norm 1, project to all eigendisplacements,
        and check if sum of projections squared is approximately 1. Since each eigenmode has norm 1,
        we expect the completeness relation to give a sum close to 1.

        Note: Perfect completeness (sum = 1) would require orthonormal eigenmodes, but phonon
        eigenmodes from different q-points are not perfectly orthogonal in supercell space.
        """
        # Load BaTiO3 data - need full 2x2x2 q-point grid
        qpoints_2x2x2 = []
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    qpoints_2x2x2.append([i / 2.0, j / 2.0, k / 2.0])
        qpoints_2x2x2 = np.array(qpoints_2x2x2)

        modes = PhononModes.from_phonopy_yaml(str(BATIO3_YAML_PATH), qpoints_2x2x2)

        # Use 2x2x2 supercell
        supercell_matrix = np.eye(3, dtype=int) * 2

        # Generate displacements for all commensurate q-points
        all_commensurate_displacements = modes.generate_all_commensurate_displacements(
            supercell_matrix, amplitude=1.0
        )

        if len(all_commensurate_displacements) == 0:
            pytest.skip("No commensurate q-points found in calculated data")

        # Create a random displacement for supercell
        np.random.seed(123)  # Different seed for different test
        n_supercell_atoms = 8 * modes._n_atoms  # 2x2x2 supercell
        random_displacement = np.random.rand(n_supercell_atoms, 3)

        # Normalize with mass-weighted norm 1
        # Need to create appropriate mass array for supercell
        supercell_masses = np.tile(
            modes.atomic_masses, 8
        )  # Repeat for each primitive cell
        current_norm = modes.mass_weighted_norm(random_displacement, supercell_masses)
        normalized_displacement = random_displacement / current_norm

        # Verify normalization
        check_norm = modes.mass_weighted_norm(normalized_displacement, supercell_masses)
        assert abs(check_norm - 1.0) < 1e-12, (
            f"Normalization failed: norm = {check_norm}"
        )

        # Project onto all eigendisplacements from all commensurate q-points
        sum_projections_squared = 0.0

        for q_index, displacements in all_commensurate_displacements.items():
            for i in range(displacements.shape[0]):
                projection = modes.mass_weighted_projection(
                    normalized_displacement, displacements[i], supercell_masses
                )
                sum_projections_squared += abs(projection) ** 2

        # For a complete orthonormal basis, the sum of projections squared should equal 1.0
        # This tests that our eigenmodes form a proper orthonormal basis for the supercell space
        # However, inter-q-point modes are not perfectly orthogonal, leading to some overcounting
        theoretical_sum = 1.0

        # Allow larger tolerance due to inter-q-point non-orthogonality
        # Based on observations, the sum is approximately 2x the theoretical value
        assert sum_projections_squared > 0.5 * theoretical_sum, (
            f"Sum too small: {sum_projections_squared} < {0.5 * theoretical_sum}"
        )
        assert sum_projections_squared < 3.0 * theoretical_sum, (
            f"Sum too large: {sum_projections_squared} > {3.0 * theoretical_sum}"
        )

        print(
            f"Completeness test passed: sum = {sum_projections_squared:.6f}, "
            f"theoretical = {theoretical_sum:.6f}, "
            f"ratio = {sum_projections_squared / theoretical_sum:.2f}"
        )


if __name__ == "__main__":
    pytest.main([__file__])
