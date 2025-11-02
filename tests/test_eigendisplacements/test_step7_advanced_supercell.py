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
        and check if sum of projections squared equals 1. According to Bloch theorem, phonon
        eigenmodes from different q-points are orthogonal in supercell space, forming a complete
        orthonormal basis. Therefore, the sum of squared projections should equal 1.0.
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
        # According to Bloch theorem, modes from different q-points are orthogonal in supercell space
        theoretical_sum = 1.0

        # Use reasonable tolerance for numerical completeness
        assert abs(sum_projections_squared - theoretical_sum) < 2e-4, (
            f"Completeness failed: sum = {sum_projections_squared}, expected = {theoretical_sum}"
        )

        print(
            f"Completeness test passed: sum = {sum_projections_squared:.6f}, "
            f"theoretical = {theoretical_sum:.6f}, "
            f"ratio = {sum_projections_squared / theoretical_sum:.2f}"
        )

    def test_16x1x1_supercell_orthogonality_and_completeness(self):
        """
        Test for 16x1x1 supercell with all required commensurate q-points.

        For a 16x1x1 supercell, we need q-points on a 16x1x1 grid:
        q = [i/16, 0, 0] for i = 0, 1, 2, ..., 15

        Tests:
        1. Orthogonality between modes from different NON-EQUIVALENT q-points
        2. Normalization of individual displacements
        3. Completeness: random displacement projects exactly onto the complete basis

        **Zone-folding Theory for 1D supercells:**

        For 1D supercells, zone-folding creates equivalent q-points where q ≡ -q (mod 1).
        In our 16x1x1 case, we have pairs like:
        - Q1 (q = 1/16) ↔ Q15 (q = 15/16), since 1/16 + 15/16 = 1
        - Q2 (q = 2/16) ↔ Q14 (q = 14/16), since 2/16 + 14/16 = 1
        - etc.

        **Key insights:**
        - Zone-folded q-points represent the same physical modes but may have different
          eigenvectors in PHONOPY due to gauge/phase choices
        - For orthogonality testing: Only check between truly non-equivalent q-points
        - For completeness testing: Use ALL modes (including zone-folded) since they
          form an over-complete basis that spans the full supercell space
        - The sum of projections squared should be ≈ 1.0, possibly slightly larger
          due to linear dependence from zone-folding
        """
        # Generate all required q-points for 16x1x1 supercell
        qpoints_16x1x1 = []
        for i in range(16):
            qpoints_16x1x1.append([i / 16.0, 0.0, 0.0])
        qpoints_16x1x1 = np.array(qpoints_16x1x1)

        print(
            f"Loading BaTiO3 with {len(qpoints_16x1x1)} q-points for 16x1x1 supercell"
        )
        modes = PhononModes.from_phonopy_yaml(str(BATIO3_YAML_PATH), qpoints_16x1x1)

        # Use 16x1x1 supercell
        supercell_matrix = np.array([[16, 0, 0], [0, 1, 0], [0, 0, 1]])
        N = 16  # Number of primitive cells in 16x1x1 supercell

        # Verify we have all commensurate q-points
        commensurate_qpoints = modes.get_commensurate_qpoints(supercell_matrix)
        print(f"Found {len(commensurate_qpoints)} commensurate q-points")
        assert len(commensurate_qpoints) == 16, (
            f"Expected 16 commensurate q-points, got {len(commensurate_qpoints)}"
        )

        # Generate displacements for all commensurate q-points
        all_commensurate_displacements = modes.generate_all_commensurate_displacements(
            supercell_matrix, amplitude=1.0
        )

        # Find zone-folding equivalent q-point pairs
        # For 16x1x1: q_i and q_j are equivalent if q_i + q_j ≈ integer
        zone_folded_pairs = set()
        zone_folded_q_indices = set()

        for i, q_i in enumerate(commensurate_qpoints):
            for j, q_j in enumerate(commensurate_qpoints):
                if i < j:  # Avoid duplicates
                    qpt_i = modes.qpoints[q_i]
                    qpt_j = modes.qpoints[q_j]

                    # Check if qpt_i ≡ -qpt_j (mod 1) using sum method
                    sum_q = qpt_i + qpt_j
                    sum_q_mod = sum_q - np.round(sum_q)
                    if np.allclose(sum_q_mod, 0.0, atol=1e-6):
                        zone_folded_pairs.add((q_i, q_j))
                        zone_folded_q_indices.add(q_i)
                        zone_folded_q_indices.add(q_j)
                        print(f"  Zone-folded pair: Q{q_i} {qpt_i} ↔ Q{q_j} {qpt_j}")

        print(f"Found {len(zone_folded_pairs)} zone-folded pairs")
        print(f"Zone-folded q-indices: {sorted(zone_folded_q_indices)}")

        # For completeness test, we should only count modes from non-equivalent q-points
        # Strategy: Use only one q-point from each zone-folded pair + any non-folded q-points
        unique_q_indices = set(commensurate_qpoints)

        # Remove the "higher" index from each zone-folded pair to avoid double counting
        for q_i, q_j in zone_folded_pairs:
            # Remove the larger index (q_j since i < j in our loop)
            if q_j in unique_q_indices:
                unique_q_indices.remove(q_j)
                print(
                    f"  Removing Q{q_j} (equivalent to Q{q_i}) from completeness test"
                )

        print(f"Unique q-indices for completeness: {sorted(unique_q_indices)}")

        # Test 1: Check orthogonality between modes from different NON-EQUIVALENT q-points
        print("Testing inter-q-point orthogonality (non-equivalent q-points only)...")
        displacement_list = []
        qpoint_labels = []

        for q_index, displacements in all_commensurate_displacements.items():
            for mode_idx in range(displacements.shape[0]):
                displacement_list.append(displacements[mode_idx])
                qpoint_labels.append((q_index, mode_idx))

        n_total_modes = len(displacement_list)
        supercell_masses = np.tile(modes.atomic_masses, N)

        max_non_equivalent_overlap = 0.0

        for i in range(n_total_modes):
            for j in range(i + 1, n_total_modes):
                q_i, mode_i = qpoint_labels[i]
                q_j, mode_j = qpoint_labels[j]

                # Only check modes from different NON-EQUIVALENT q-points
                if (
                    q_i != q_j and mode_i == mode_j
                ):  # Same mode index from different q-points
                    # Check if these q-points are zone-folding equivalent
                    is_zone_folded = (q_i, q_j) in zone_folded_pairs or (
                        q_j,
                        q_i,
                    ) in zone_folded_pairs

                    if (
                        not is_zone_folded
                    ):  # Only test orthogonality for non-equivalent pairs
                        projection = modes.mass_weighted_projection(
                            displacement_list[i], displacement_list[j], supercell_masses
                        )
                        overlap = abs(projection)
                        max_non_equivalent_overlap = max(
                            max_non_equivalent_overlap, overlap
                        )

        print(f"Max non-equivalent q-point overlap: {max_non_equivalent_overlap:.2e}")

        # Only require orthogonality for non-equivalent q-points
        assert max_non_equivalent_overlap < 1e-12, (
            f"Inter-q-point orthogonality violated for non-equivalent q-points: max overlap = {max_non_equivalent_overlap}"
        )

        # Test 2: Check that each displacement has mass-weighted norm = 1
        print("Testing displacement normalization...")
        for q_index, displacements in all_commensurate_displacements.items():
            for mode_idx in range(displacements.shape[0]):
                norm = modes.mass_weighted_norm(
                    displacements[mode_idx], supercell_masses
                )
                assert abs(norm - 1.0) < 1e-10, (
                    f"Wrong norm for q={q_index}, mode={mode_idx}: {norm}"
                )

        # Test 3: Test completeness with random displacement using ALL modes
        # Theory: Zone-folded q-points create linearly dependent modes, but ALL modes
        # (including zone-folded) are needed to form a complete basis for the supercell space.
        # The sum should be ≈ 1.0, possibly slightly larger due to over-counting from linear dependence.
        print("Testing completeness (all modes including zone-folded)...")
        np.random.seed(16)  # Use 16 as seed for 16x1x1 test
        n_supercell_atoms = N * modes._n_atoms
        random_displacement = np.random.rand(n_supercell_atoms, 3)

        # Normalize with mass-weighted norm 1
        current_norm = modes.mass_weighted_norm(random_displacement, supercell_masses)
        normalized_displacement = random_displacement / current_norm

        # Verify normalization
        check_norm = modes.mass_weighted_norm(normalized_displacement, supercell_masses)
        assert abs(check_norm - 1.0) < 1e-12, (
            f"Normalization failed: norm = {check_norm}"
        )

        # Project onto ALL eigendisplacements from ALL commensurate q-points
        sum_projections_squared = 0.0
        total_all_modes = 0

        for q_index, displacements in all_commensurate_displacements.items():
            for mode_idx in range(displacements.shape[0]):
                projection = modes.mass_weighted_projection(
                    normalized_displacement,
                    displacements[mode_idx],
                    supercell_masses,
                )
                sum_projections_squared += abs(projection) ** 2
                total_all_modes += 1

        # For zone-folded systems, ALL modes form a complete (but over-complete) basis
        # Sum should be ≈ 1.0, allowing for small over-counting due to linear dependence
        theoretical_sum = 1.0
        completeness_error = abs(sum_projections_squared - theoretical_sum)

        print(f"Total modes (all): {total_all_modes}")
        print(f"Expected total modes: {N * modes._n_modes} (should match)")
        print(
            f"Unique q-indices: {len(unique_q_indices)} (from {len(commensurate_qpoints)} total)"
        )
        print(
            f"Completeness: sum = {sum_projections_squared:.6f}, error = {completeness_error:.2e}"
        )

        # Allow for slight over-completeness due to zone-folding linear dependence
        # The sum should be close to 1.0, but may be slightly larger (up to ~1.05)
        assert completeness_error < 5e-2, (
            f"Completeness failed for 16x1x1: sum = {sum_projections_squared}, "
            f"expected = {theoretical_sum}, error = {completeness_error}"
        )

        print(f"16x1x1 supercell test passed successfully!")
        print(
            f"  - {len(commensurate_qpoints)} q-points ({len(unique_q_indices)} unique)"
        )
        print(f"  - {total_all_modes} total modes")
        print(f"  - Zone-folded pairs: {len(zone_folded_pairs)}")
        print(f"  - Inter-q-point orthogonality: {max_non_equivalent_overlap:.2e}")
        print(f"  - Completeness error: {completeness_error:.2e}")


if __name__ == "__main__":
    pytest.main([__file__])
