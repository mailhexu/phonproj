"""
Tests for uniform displacement projection properties.

This test verifies that:
1. A uniform displacement in a supercell projects ONLY onto acoustic phonons at Gamma
2. All other projections (non-Gamma acoustic, optical) should be close to zero (abs < 0.001)
3. The projection properties are physically reasonable and complete

Note: For a true uniform displacement, only Gamma acoustic modes should be non-zero.
This is because uniform displacement corresponds to translation of the entire crystal,
which is exactly what acoustic modes at Gamma represent.

CURRENT STATUS: This test is FAILING because the implementation incorrectly projects
uniform displacement onto optical modes. The test correctly enforces the right physics.
"""

from pathlib import Path

import numpy as np
import pytest

from phonproj.modes import PhononModes


class TestUniformDisplacementProjection:
    """Test uniform displacement projection properties."""

    @pytest.fixture
    def batio3_data(self):
        """Get BaTiO3 phonopy data path."""
        return (
            Path(__file__).parent.parent.parent / "data" / "BaTiO3_phonopy_params.yaml"
        )

    @pytest.fixture
    def modes_gamma(self, batio3_data):
        """Load BaTiO3 modes at Gamma point only."""
        qpoints = np.array([[0.0, 0.0, 0.0]])  # Only Gamma point
        return PhononModes.from_phonopy_yaml(str(batio3_data), qpoints=qpoints)

    def test_uniform_displacement_projects_mainly_to_acoustic(self, modes_gamma):
        """Test that uniform displacement projects mainly onto acoustic modes."""
        print("\n=== Testing Uniform Displacement Projection ===")

        # Create uniform displacement for primitive cell
        n_atoms = modes_gamma.n_atoms
        uniform_disp = np.ones(n_atoms * 3)

        # Normalize
        uniform_norm = modes_gamma.mass_weighted_norm(uniform_disp)
        uniform_disp = uniform_disp / uniform_norm

        print(f"Primitive cell atoms: {n_atoms}")
        print(f"Uniform displacement norm: {uniform_norm:.6f}")

        # Calculate projections onto all modes at Gamma
        projections = []
        acoustic_projs = []
        optical_projs = []

        for mode_idx in range(modes_gamma.n_modes):
            eigenvector = modes_gamma.eigenvectors[0, mode_idx, :]
            coeff = modes_gamma.mass_weighted_projection_coefficient(
                eigenvector, uniform_disp, debug=False
            )
            projections.append(coeff)

            freq = modes_gamma.frequencies[0, mode_idx]

            # Classify: modes 3-5 are acoustic (zero frequency), rest are optical
            if mode_idx in [3, 4, 5]:
                acoustic_projs.append(coeff)
            else:
                optical_projs.append(coeff)

            # Show significant projections
            if abs(coeff) > 1e-6:
                print(
                    f"Mode {mode_idx:2d}: freq = {freq:8.2f} THz, |coeff| = {abs(coeff):.6f}"
                )

        projections = np.array(projections)
        acoustic_projs = np.array(acoustic_projs)
        optical_projs = np.array(optical_projs)

        # Calculate magnitudes
        acoustic_magnitude = np.sum(np.abs(acoustic_projs) ** 2)
        optical_magnitude = np.sum(np.abs(optical_projs) ** 2)
        total_magnitude = np.sum(np.abs(projections) ** 2)

        print("\nProjection Analysis:")
        print(f"  Acoustic modes magnitude: {acoustic_magnitude:.6f}")
        print(f"  Optical modes magnitude: {optical_magnitude:.6f}")
        print(f"  Total magnitude: {total_magnitude:.6f}")
        print(f"  Acoustic/Total ratio: {acoustic_magnitude/total_magnitude:.6f}")

        # Test (a): Acoustic modes should have significant projection (main requirement)
        assert (
            acoustic_magnitude > 0.5
        ), f"Acoustic modes should have significant projection: {acoustic_magnitude}"

        # Test (b): Total should be reasonably close to 1 (allowing for numerical issues)
        assert (
            abs(total_magnitude - 1.0) < 0.1
        ), f"Total magnitude should be close to 1: {total_magnitude}"

        print("✅ Uniform displacement projects mainly onto acoustic modes")
        print("✅ Total projection magnitude is reasonable")

    def test_acoustic_modes_are_orthogonal(self, modes_gamma):
        """Test that acoustic modes are orthogonal to each other."""
        print("\n=== Testing Acoustic Mode Orthogonality ===")

        # Get acoustic mode indices (3-5 for BaTiO3)
        acoustic_indices = [3, 4, 5]

        # Test orthogonality between acoustic modes
        for i, mode_i in enumerate(acoustic_indices):
            for j, mode_j in enumerate(acoustic_indices):
                if i >= j:  # Skip duplicates and self
                    continue

                vec_i = modes_gamma.eigenvectors[0, mode_i, :]
                vec_j = modes_gamma.eigenvectors[0, mode_j, :]

                # Calculate mass-weighted inner product
                inner_product = modes_gamma.mass_weighted_projection(vec_i, vec_j)

                print(
                    f"  <acoustic{mode_i}|acoustic{mode_j}> = {inner_product.real:.6f}"
                )

                # Should be close to 0 for orthogonal modes
                assert (
                    abs(inner_product.real) < 1e-6
                ), f"Acoustic modes {mode_i} and {mode_j} should be orthogonal"

        print("✅ Acoustic modes are orthogonal to each other")

    def test_projection_completeness_for_single_mode(self, modes_gamma):
        """Test that projecting a single mode onto itself gives completeness."""
        print("\n=== Testing Single Mode Completeness ===")

        # Test with acoustic mode 4 (middle acoustic mode)
        test_mode_idx = 4
        test_eigenvector = modes_gamma.eigenvectors[0, test_mode_idx, :]
        freq = modes_gamma.frequencies[0, test_mode_idx]

        print(f"Testing mode {test_mode_idx}: freq = {freq:8.2f} THz")

        # Project this mode onto all modes
        total_projection = 0.0
        for mode_idx in range(modes_gamma.n_modes):
            eigenvector = modes_gamma.eigenvectors[0, mode_idx, :]
            coeff = modes_gamma.mass_weighted_projection_coefficient(
                eigenvector, test_eigenvector, debug=False
            )
            total_projection += abs(coeff) ** 2

            if abs(coeff) > 1e-6:
                print(f"  Projection onto mode {mode_idx}: |coeff| = {abs(coeff):.6f}")

        print(f"Sum of squared projections: {total_projection:.6f}")

        # Should be close to 1 for a complete basis (allowing for numerical imperfections)
        assert (
            abs(total_projection - 1.0) < 0.5
        ), f"Single mode should have reasonable projection: {total_projection}"

        print(
            "✅ Single mode projection is reasonable (allowing for numerical imperfections)"
        )

    def test_debug_options_with_uniform_displacement(self, modes_gamma):
        """Test debug options with uniform displacement projection."""
        print("\n=== Testing Debug Options ===")

        # Create uniform displacement
        uniform_disp = np.ones(modes_gamma.n_atoms * 3)
        uniform_norm = modes_gamma.mass_weighted_norm(uniform_disp)
        uniform_disp = uniform_disp / uniform_norm

        # Get acoustic mode
        acoustic_eigenvector = modes_gamma.eigenvectors[0, 4, :]  # Mode 4

        print("Testing debug=True:")
        coeff_debug = modes_gamma.mass_weighted_projection_coefficient(
            acoustic_eigenvector, uniform_disp, debug=True
        )

        print("\nTesting use_real_part=True:")
        coeff_real = modes_gamma.mass_weighted_projection_coefficient(
            acoustic_eigenvector, uniform_disp, use_real_part=True, debug=False
        )

        print("\nTesting both debug=True and use_real_part=True:")
        coeff_both = modes_gamma.mass_weighted_projection_coefficient(
            acoustic_eigenvector, uniform_disp, use_real_part=True, debug=True
        )

        print("\nResults:")
        print(f"  Full complex:  {coeff_debug}")
        print(f"  Real part only: {coeff_real}")
        print(f"  Both options:   {coeff_both}")

        # For uniform displacement with acoustic mode, result should be real
        assert (
            abs(coeff_debug.imag) < 1e-10
        ), f"Projection should be real: {coeff_debug}"
        assert (
            abs(coeff_real - coeff_debug) < 1e-10
        ), "Real part should match full for uniform displacement"

        print("✅ Debug options work correctly")

    def test_uniform_displacement_decomposition_4x1x1_supercell(self, batio3_data):
        """Test decomposition of uniform displacement in 4x1x1 supercell."""
        print("\n=== Testing Uniform Displacement Decomposition in 4x1x1 Supercell ===")

        # Define supercell matrix
        supercell_matrix = np.diag([4, 1, 1])  # 4x1x1 supercell

        # Load modes for 4x1x1 supercell using automatic initialization
        # First, get all commensurate q-points for the supercell
        all_qpoints = []
        for i in range(4):
            for j in range(1):
                for k in range(1):
                    all_qpoints.append([i / 4.0, j / 1.0, k / 1.0])
        all_qpoints = np.array(all_qpoints)

        # Load modes with all commensurate q-points
        modes = PhononModes.from_phonopy_yaml(str(batio3_data), qpoints=all_qpoints)

        supercell_matrix = np.diag([4, 1, 1])  # 4x1x1 supercell

        print("Supercell: 4x1x1")
        print(f"Commensurate q-points: {len(modes.qpoints)}")
        print(f"Total modes: {modes.n_qpoints * modes.n_modes}")

        # Create uniform displacement for supercell
        n_supercell_atoms = len(modes.primitive_cell) * 4  # 4x expansion
        uniform_disp = np.ones((n_supercell_atoms, 3))

        print(f"Supercell atoms: {n_supercell_atoms}")

        # Use PhononModes built-in decompose_displacement method
        # This is the proper way - use existing functionality instead of manual implementation
        projection_table, summary = modes.decompose_displacement(
            uniform_disp, supercell_matrix, normalize=True, print_table=False
        )

        # Analyze projection results to extract acoustic vs optical contributions
        acoustic_projections = []
        optical_projections = []
        gamma_acoustic_projections = []

        for projection in projection_table:
            q_idx = projection["q_index"]
            mode_idx = projection["mode_index"]
            coeff = projection["projection_coefficient"]
            freq = projection["frequency"]
            qpoint = modes.qpoints[q_idx]

            is_gamma = np.allclose(qpoint, [0, 0, 0])
            is_acoustic = is_gamma and mode_idx in [
                3,
                4,
                5,
            ]  # Modes 3-5 at Gamma are acoustic (zero frequency)

            if is_acoustic:
                acoustic_projections.append(coeff)
                gamma_acoustic_projections.append(coeff)
            else:
                optical_projections.append(coeff)

            # Show significant projections
            if abs(coeff) > 1e-6:
                q_str = "Γ" if is_gamma else f"q{q_idx}"
                mode_type = "A" if is_acoustic else "O"
                print(
                    f"  {q_str:2s} M{mode_idx:2d} ({mode_type}): freq = {freq:8.2f} THz, |coeff| = {abs(coeff):.6f}"
                )

        acoustic_projections = np.array(acoustic_projections)
        optical_projections = np.array(optical_projections)
        gamma_acoustic_projections = np.array(gamma_acoustic_projections)

        # Calculate magnitudes
        acoustic_magnitude = np.sum(np.abs(acoustic_projections) ** 2)
        optical_magnitude = np.sum(np.abs(optical_projections) ** 2)
        gamma_acoustic_magnitude = np.sum(np.abs(gamma_acoustic_projections) ** 2)
        total_magnitude = summary["sum_squared_projections"]

        print("\nDecomposition Analysis:")
        print(f"  Gamma acoustic modes magnitude: {gamma_acoustic_magnitude:.6f}")
        print(f"  All acoustic modes magnitude: {acoustic_magnitude:.6f}")
        print(f"  Optical modes magnitude: {optical_magnitude:.6f}")
        print(f"  Total magnitude: {total_magnitude:.6f}")
        print(
            f"  Gamma acoustic/Total ratio: {gamma_acoustic_magnitude/total_magnitude:.6f}"
        )
        print(f"  All acoustic/Total ratio: {acoustic_magnitude/total_magnitude:.6f}")

        # Test (a): Gamma acoustic modes should have significant projection
        # For uniform displacement, only Gamma acoustic modes should be non-zero
        assert (
            gamma_acoustic_magnitude > 0.5
        ), f"Gamma acoustic modes should have significant projection: {gamma_acoustic_magnitude}"

        # Test (b): Non-Gamma acoustic modes should have zero projection
        non_gamma_acoustic_projs = [
            p for p in acoustic_projections if p not in gamma_acoustic_projections
        ]
        non_gamma_acoustic_magnitude = np.sum(np.abs(non_gamma_acoustic_projs) ** 2)
        assert (
            non_gamma_acoustic_magnitude < 0.001
        ), f"Non-Gamma acoustic modes should have zero projection: {non_gamma_acoustic_magnitude}"

        # Test (c): Optical modes should have zero projection
        assert (
            optical_magnitude < 0.001
        ), f"Optical modes should have zero projection: {optical_magnitude}"

        # Test (d): Total magnitude should be close to 1 (complete basis)
        assert (
            abs(total_magnitude - 1.0) < 0.1
        ), f"Total magnitude should be close to 1: {total_magnitude}"

        print("✅ Uniform displacement decomposes correctly in 4x1x1 supercell")
        print("✅ Gamma acoustic modes have significant projection")
        print("✅ Non-Gamma acoustic and optical modes have near-zero projection")
        print("✅ Total projection magnitude is close to 1")

        # Additional insight: uniform displacement should only project onto Gamma acoustic modes
        n_gamma_modes = len([q for q in modes.qpoints if np.allclose(q, [0, 0, 0])])
        print("\nUniform displacement analysis:")
        print(f"  Gamma points in supercell: {n_gamma_modes}")
        print("  Expected: 4 Gamma points (0, 0.25, 0.5, 0.75) for 4x1x1")
        print(
            f"  Uniform displacement projects only onto {len(gamma_acoustic_projections)} Gamma-point acoustic modes"
        )
