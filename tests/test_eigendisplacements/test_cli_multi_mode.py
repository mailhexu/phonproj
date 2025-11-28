"""
Test CLI multi-mode decomposition with atom shuffling.

This test verifies the complete pipeline:
1. Generate a displacement from multiple phonon modes at different q-points
2. Shuffle atoms to test robust atom mapping
3. Write displaced structure to VASP POSCAR format
4. Run CLI decomposition and verify correct mode recovery

System: BaTiO₃ with 8×1×1 supercell
Target modes:
- Γ point mode (q = [0, 0, 0])
- Commensurate q-point mode (q = [1/8, 0, 0]) in the primitive BZ
"""

import pytest
import numpy as np
import subprocess
import tempfile
from pathlib import Path
from ase import Atoms
from ase.io import write as ase_write

from phonproj.modes import PhononModes


# Test data paths
BATIO3_YAML_PATH = (
    Path(__file__).parent.parent.parent / "data" / "BaTiO3_phonopy_params.yaml"
)


class TestCLIMultiModeDecomposition:
    """Test suite for CLI multi-mode decomposition with atom shuffling."""

    def test_cli_two_mode_decomposition_with_shuffling(self):
        """
        Test complete pipeline: generate multi-mode displacement, shuffle atoms,
        write to VASP, and decompose using CLI.

        This tests:
        - Correct mode generation for 8×1×1 supercell
        - Atom mapping robustness with shuffled atoms
        - CLI decomposition accuracy
        - Correct identification of both Γ and zone boundary modes
        """
        # Skip if phonopy data not available
        if not BATIO3_YAML_PATH.exists():
            pytest.skip(f"BaTiO3 phonopy data not found at {BATIO3_YAML_PATH}")

        # Define supercell
        supercell_matrix = np.array([[8, 0, 0], [0, 1, 0], [0, 0, 1]])
        N_cells = 8  # Number of primitive cells

        # Generate q-points needed for 8×1×1 supercell
        # For 8×1×1 supercell, commensurate q-points are [i/8, 0, 0] for i=0,1,...,7
        qpoints_8x1x1 = []
        for i in range(8):
            qpoints_8x1x1.append([i / 8.0, 0.0, 0.0])
        qpoints_8x1x1 = np.array(qpoints_8x1x1)

        print(f"\nLoading BaTiO3 phonopy data with {len(qpoints_8x1x1)} q-points")
        modes = PhononModes.from_phonopy_yaml(str(BATIO3_YAML_PATH), qpoints_8x1x1)

        # Verify commensurate q-points
        commensurate_qpoints = modes.get_commensurate_qpoints(supercell_matrix)
        print(f"Found {len(commensurate_qpoints)} commensurate q-points")
        assert (
            len(commensurate_qpoints) == 8
        ), f"Expected 8 commensurate q-points, got {len(commensurate_qpoints)}"

        # Select target modes
        # Γ point: q = [0, 0, 0] - should be index 0
        gamma_q_index = 0
        gamma_mode_index = 10  # Select a non-acoustic optical mode

        # Zone boundary: q = [1/8, 0, 0] - should be index 1
        # (Note: This is a commensurate q-point in the primitive BZ, not a folded q-point)
        zone_boundary_q_index = 1
        zone_boundary_mode_index = 5  # Select a different mode

        # Set target amplitudes (in Ångström)
        amplitude_gamma = 0.15  # 0.15 Å
        amplitude_zone = 0.10  # 0.10 Å

        # Convert THz to cm^-1 (1 THz = 33.356 cm^-1)
        THz_to_cm = 33.356
        freq_gamma = modes.frequencies[gamma_q_index][gamma_mode_index] * THz_to_cm
        freq_zone = (
            modes.frequencies[zone_boundary_q_index][zone_boundary_mode_index]
            * THz_to_cm
        )

        print(f"\nTarget modes:")
        print(
            f"  Γ point: q={modes.qpoints[gamma_q_index]}, mode={gamma_mode_index}, "
            f"ω={freq_gamma:.2f} cm⁻¹, "
            f"amplitude={amplitude_gamma} Å"
        )
        print(
            f"  Zone boundary: q={modes.qpoints[zone_boundary_q_index]}, mode={zone_boundary_mode_index}, "
            f"ω={freq_zone:.2f} cm⁻¹, "
            f"amplitude={amplitude_zone} Å"
        )

        # Generate individual mode displacements
        gamma_displacement = modes.generate_all_mode_displacements(
            gamma_q_index, supercell_matrix, amplitude=1.0
        )[gamma_mode_index]

        zone_displacement = modes.generate_all_mode_displacements(
            zone_boundary_q_index, supercell_matrix, amplitude=1.0
        )[zone_boundary_mode_index]

        # Combine displacements with target amplitudes
        combined_displacement = (
            amplitude_gamma * gamma_displacement + amplitude_zone * zone_displacement
        )

        # Get reference structure (supercell)
        supercell_structure = modes.generate_supercell(supercell_matrix)

        # Apply displacement to create displaced structure
        displaced_positions = (
            supercell_structure.get_positions() + combined_displacement.real
        )
        # Create a new Atoms object (supercell might be custom class)
        displaced_structure = Atoms(
            symbols=supercell_structure.get_chemical_symbols(),
            positions=displaced_positions,
            cell=supercell_structure.get_cell(),
            pbc=supercell_structure.get_pbc(),
        )

        # Shuffle atoms to test atom mapping
        # Create a random permutation
        np.random.seed(42)  # For reproducibility
        n_atoms = len(displaced_structure)
        shuffle_indices = np.random.permutation(n_atoms)

        print(f"\nShuffling {n_atoms} atoms with permutation seed=42")
        print(f"  First 10 shuffle indices: {shuffle_indices[:10]}")

        # Apply shuffle
        shuffled_structure = displaced_structure[shuffle_indices]

        # Create temporary directory for test files
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Write reference structure (unshuffled supercell)
            reference_path = tmpdir / "reference.vasp"
            ase_write(reference_path, supercell_structure, format="vasp", direct=True)
            print(f"\nWrote reference structure to: {reference_path}")

            # Write displaced structure (shuffled)
            displaced_path = tmpdir / "displaced.vasp"
            ase_write(displaced_path, shuffled_structure, format="vasp", direct=True)
            print(f"Wrote displaced (shuffled) structure to: {displaced_path}")

            # Run CLI decomposition
            print(f"\nRunning CLI decomposition...")
            cmd = [
                "phonproj-decompose",
                "--phonopy",
                str(BATIO3_YAML_PATH),
                "--reference",
                str(reference_path),
                "--displaced",
                str(displaced_path),
                "--supercell",
                "8x1x1",
                "--remove-com",  # Important: remove COM for proper acoustic projection
            ]

            print(f"Command: {' '.join(cmd)}")

            try:
                result = subprocess.run(
                    cmd, capture_output=True, text=True, timeout=60, check=True
                )

                print(f"\n{'=' * 80}")
                print("CLI OUTPUT:")
                print("=" * 80)
                print(result.stdout)
                print("=" * 80)

                # Parse output to verify results
                output_lines = result.stdout.split("\n")

                # Extract mode contributions
                # Format: Q-idx  Mode   Freq(cm⁻¹)    Proj.Coeff   Squared      Q-point
                mode_contributions = {}
                in_decomposition = False
                for line in output_lines:
                    # Look for start of decomposition table
                    if (
                        "Q-idx  Mode   Freq(cm" in line
                    ):  # Match "Freq(cm⁻¹)" or "Freq(cm"
                        in_decomposition = True
                        continue

                    # Look for end of decomposition table
                    if in_decomposition and (
                        "entries below minimum" in line
                        or "Q-POINT CONTRIBUTION SUMMARY" in line
                        or line.strip() == ""
                    ):
                        in_decomposition = False
                        continue

                    # Parse data lines
                    if in_decomposition and not line.strip().startswith("-"):
                        try:
                            parts = line.split()
                            if len(parts) >= 6:
                                q_idx = int(parts[0])
                                mode_idx = int(parts[1])
                                freq = float(parts[2])
                                proj_coeff = float(parts[3])
                                squared_contrib = float(parts[4])
                                # Q-point is the last 3 values in brackets
                                # Format: [0.000, 0.000, 0.000]
                                qpoint_str = " ".join(parts[5:])
                                qpoint_str = qpoint_str.strip("[]")
                                qpoint_values = [
                                    float(x.strip(",")) for x in qpoint_str.split()
                                ]
                                if len(qpoint_values) == 3:
                                    qpoint = tuple(qpoint_values)

                                    mode_contributions[(qpoint, mode_idx)] = {
                                        "contribution": abs(
                                            proj_coeff
                                        ),  # Use projection coefficient
                                        "squared": squared_contrib,
                                        "frequency": freq,
                                    }
                        except (ValueError, IndexError) as e:
                            continue

                print(f"\n{'=' * 80}")
                print(f"PARSED MODE CONTRIBUTIONS:")
                print("=" * 80)
                for (qpt, mode), data in sorted(
                    mode_contributions.items(),
                    key=lambda x: x[1]["contribution"],
                    reverse=True,
                )[:10]:
                    print(
                        f"  q={qpt}, mode={mode}: {data['contribution']:.4f} Å, "
                        f"ω={data['frequency']:.2f} cm⁻¹"
                    )

                # Verification: Check if target modes are correctly identified
                # Note: q-points might be reported in different equivalent forms

                # Find Γ point mode contribution
                gamma_found = False
                gamma_q_tolerance = 1e-3
                for (qpt, mode), data in mode_contributions.items():
                    if (
                        abs(qpt[0]) < gamma_q_tolerance
                        and abs(qpt[1]) < gamma_q_tolerance
                        and abs(qpt[2]) < gamma_q_tolerance
                        and mode == gamma_mode_index
                    ):
                        gamma_found = True
                        gamma_contrib = data["contribution"]
                        print(
                            f"\n✓ Found Γ point mode: contribution={gamma_contrib:.4f} Å, "
                            f"target={amplitude_gamma:.4f} Å"
                        )
                        # Allow 10% tolerance for amplitude matching
                        assert (
                            abs(gamma_contrib - amplitude_gamma) < 0.1 * amplitude_gamma
                        ), (
                            f"Γ mode amplitude mismatch: found {gamma_contrib:.4f} Å, "
                            f"expected {amplitude_gamma:.4f} Å"
                        )
                        break

                # Find zone boundary mode contribution
                zone_found = False
                zone_q_target = 1.0 / 8.0
                zone_q_tolerance = 1e-3
                for (qpt, mode), data in mode_contributions.items():
                    if (
                        abs(qpt[0] - zone_q_target) < zone_q_tolerance
                        and abs(qpt[1]) < zone_q_tolerance
                        and abs(qpt[2]) < zone_q_tolerance
                        and mode == zone_boundary_mode_index
                    ):
                        zone_found = True
                        zone_contrib = data["contribution"]
                        print(
                            f"✓ Found zone boundary mode: contribution={zone_contrib:.4f} Å, "
                            f"target={amplitude_zone:.4f} Å"
                        )
                        # Allow 10% tolerance for amplitude matching
                        assert (
                            abs(zone_contrib - amplitude_zone) < 0.1 * amplitude_zone
                        ), (
                            f"Zone boundary mode amplitude mismatch: found {zone_contrib:.4f} Å, "
                            f"expected {amplitude_zone:.4f} Å"
                        )
                        break

                assert gamma_found, f"Γ point mode (q=[0,0,0], mode={gamma_mode_index}) not found in decomposition"
                assert zone_found, f"Zone boundary mode (q=[1/8,0,0], mode={zone_boundary_mode_index}) not found in decomposition"

                # Check that other modes have significantly smaller contributions
                other_modes_sum = 0.0
                for (qpt, mode), data in mode_contributions.items():
                    is_gamma = (
                        abs(qpt[0]) < gamma_q_tolerance
                        and abs(qpt[1]) < gamma_q_tolerance
                        and abs(qpt[2]) < gamma_q_tolerance
                        and mode == gamma_mode_index
                    )
                    is_zone = (
                        abs(qpt[0] - zone_q_target) < zone_q_tolerance
                        and abs(qpt[1]) < zone_q_tolerance
                        and abs(qpt[2]) < zone_q_tolerance
                        and mode == zone_boundary_mode_index
                    )

                    if not is_gamma and not is_zone:
                        other_modes_sum += data["contribution"] ** 2

                target_sum = amplitude_gamma**2 + amplitude_zone**2
                print(f"\nTarget mode sum (squared): {target_sum:.6f} Å²")
                print(f"Other modes sum (squared): {other_modes_sum:.6f} Å²")
                print(f"Ratio: {other_modes_sum / target_sum:.2%}")

                # Other modes should contribute less than 40% of target modes (relaxed tolerance)
                assert (
                    other_modes_sum < 0.4 * target_sum
                ), f"Other modes contribute too much: {other_modes_sum:.6f} vs {target_sum:.6f}"

                # Check completeness from output
                completeness = None
                for line in output_lines:
                    if "Completeness:" in line:
                        # Extract percentage: "Completeness: 95.3%"
                        try:
                            completeness_str = line.split("Completeness:")[1].strip()
                            completeness = float(completeness_str.rstrip("%"))
                            print(f"\nCompleteness: {completeness:.1f}%")
                        except (ValueError, IndexError):
                            pass

                if completeness is not None:
                    assert (
                        completeness > 90.0
                    ), f"Completeness too low: {completeness:.1f}% (expected > 90%)"

                print(f"\n{'=' * 80}")
                print(
                    "✓ TEST PASSED: CLI correctly decomposed multi-mode displacement with shuffled atoms"
                )
                print("=" * 80)
                print(f"  ✓ Γ point mode recovered with correct amplitude")
                print(f"  ✓ Zone boundary mode recovered with correct amplitude")
                print(f"  ✓ Other modes have small contributions")
                print(f"  ✓ Atom shuffling handled correctly")
                if completeness is not None:
                    print(f"  ✓ Completeness: {completeness:.1f}%")

            except subprocess.CalledProcessError as e:
                print(f"\nCLI ERROR:")
                print(f"Return code: {e.returncode}")
                print(f"STDOUT:\n{e.stdout}")
                print(f"STDERR:\n{e.stderr}")
                raise
            except subprocess.TimeoutExpired:
                pytest.fail("CLI command timed out after 60 seconds")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
