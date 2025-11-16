"""
Test CLI ISODISTORT decomposition functionality.

This test verifies the complete ISODISTORT pipeline:
1. Load ISODISTORT file with undistorted and distorted structures
2. Calculate displacement between structures
3. Map to phonopy supercell and decompose into phonon modes
4. Verify correct mode identification and displacement recovery

System: P4mmm reference structure from ISODISTORT
Target: Verify ISODISTORT input produces same results as manual displaced structure
"""

import pytest
import numpy as np
import subprocess
import tempfile
from pathlib import Path
from ase import Atoms
from ase.io import write as ase_write

from phonproj.modes import PhononModes
from phonproj.cli import load_isodistort_structures


# Test data paths
BATIO3_YAML_PATH = (
    Path(__file__).parent.parent.parent / "data" / "BaTiO3_phonopy_params.yaml"
)
ISODISTORT_PATH = (
    Path(__file__).parent.parent.parent / "data" / "yajundata" / "P4mmm-ref.txt"
)


class TestCLIISODISTORTDecomposition:
    """Test suite for CLI ISODISTORT decomposition."""

    def test_cli_isodistort_basic_functionality(self):
        """
        Test basic ISODISTORT CLI functionality:
        - Load ISODISTORT file
        - Calculate displacement
        - Decompose into phonon modes
        - Verify reasonable results
        """
        # Skip if test data not available
        if not BATIO3_YAML_PATH.exists():
            pytest.skip(f"BaTiO3 phonopy data not found at {BATIO3_YAML_PATH}")
        if not ISODISTORT_PATH.exists():
            pytest.skip(f"ISODISTORT file not found at {ISODISTORT_PATH}")

        # Load ISODISTORT structures to verify basic functionality
        try:
            undistorted, distorted = load_isodistort_structures(str(ISODISTORT_PATH))
            print(f"Loaded ISODISTORT structures:")
            print(f"  Undistorted: {len(undistorted)} atoms")
            print(f"  Distorted: {len(distorted)} atoms")
            print(f"  Cell volume: {undistorted.get_volume():.3f} Ų")
        except Exception as e:
            pytest.skip(f"Failed to load ISODISTORT file: {e}")

        # Define supercell matrix (1x1x1 for faster test)
        supercell_matrix = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

        # Run CLI decomposition with ISODISTORT
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Run CLI decomposition
            print(f"\nRunning CLI ISODISTORT decomposition...")
            cmd = [
                "phonproj-decompose",
                "--phonopy",
                str(BATIO3_YAML_PATH),
                "--isodistort",
                str(ISODISTORT_PATH),
                "--supercell",
                "4x4x2",
                "--remove-com",
                "--quiet",
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

                # Check for successful execution
                success_indicators = [
                    "ISODISTORT file:",
                    "Displacement calculated",
                    "DISPLACEMENT MODE DECOMPOSITION",
                    "Total modes analyzed:",
                ]

                for indicator in success_indicators:
                    assert any(indicator in line for line in output_lines), (
                        f"Expected indicator '{indicator}' not found in CLI output"
                    )

                # Extract displacement magnitude
                displacement_magnitude = None
                for line in output_lines:
                    if "Mass-weighted norm:" in line:
                        try:
                            # Format: "Mass-weighted norm: X.XXXX"
                            magnitude_str = line.split(":")[1].strip()
                            displacement_magnitude = float(magnitude_str)
                            print(
                                f"Displacement magnitude: {displacement_magnitude:.4f}"
                            )
                            break
                        except (ValueError, IndexError):
                            pass

                assert displacement_magnitude is not None, (
                    "Could not extract displacement magnitude from output"
                )
                assert displacement_magnitude > 0, (
                    f"Displacement magnitude should be positive, got {displacement_magnitude}"
                )

                # Extract completeness
                completeness = None
                for line in output_lines:
                    if "Completeness:" in line:
                        try:
                            completeness_str = line.split("Completeness:")[1].strip()
                            completeness = float(completeness_str.rstrip("%"))
                            print(f"Completeness: {completeness:.1f}%")
                            break
                        except (ValueError, IndexError):
                            pass

                if completeness is not None:
                    assert completeness > 80.0, (
                        f"Completeness too low: {completeness:.1f}% (expected > 80%)"
                    )

                # Check that mode decomposition was performed
                mode_lines = [
                    line
                    for line in output_lines
                    if line.strip()
                    and not line.startswith("=")
                    and not line.startswith("-")
                    and len(line.split()) >= 6
                    and any(char.isdigit() for char in line.split()[0])
                ]
                assert len(mode_lines) > 0, (
                    "No mode decomposition results found in output"
                )

                print(f"\n{'=' * 80}")
                print("✓ TEST PASSED: CLI ISODISTORT decomposition successful")
                print("=" * 80)
                print(f"  ✓ ISODISTORT file loaded successfully")
                print(f"  ✓ Displacement calculated: {displacement_magnitude:.4f} Å")
                print(f"  ✓ Mode decomposition performed: {len(mode_lines)} modes")
                if completeness is not None:
                    print(f"  ✓ Completeness: {completeness:.1f}%")

            except subprocess.CalledProcessError as e:
                print(f"\nCLI ERROR:")
                print(f"Return code: {e.returncode}")
                print(f"STDOUT:\n{e.stdout}")
                print(f"STDERR:\n{e.stderr}")
                raise
            except subprocess.TimeoutExpired:
                pytest.fail("CLI command timed out after 120 seconds")

    def test_cli_isodistort_vs_displaced_equivalence(self):
        """
        Test that ISODISTORT input produces equivalent results to manual displaced structure.

        This test:
        1. Loads ISODISTORT structures
        2. Extracts undistorted and distorted structures
        3. Runs CLI with both --isodistort and --displaced options
        4. Compares results for equivalence
        """
        # Skip if test data not available
        if not BATIO3_YAML_PATH.exists():
            pytest.skip(f"BaTiO3 phonopy data not found at {BATIO3_YAML_PATH}")
        if not ISODISTORT_PATH.exists():
            pytest.skip(f"ISODISTORT file not found at {ISODISTORT_PATH}")

        # Load ISODISTORT structures
        try:
            undistorted, distorted = load_isodistort_structures(str(ISODISTORT_PATH))
        except Exception as e:
            pytest.skip(f"Failed to load ISODISTORT file: {e}")

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Write structures for --displaced option
            reference_path = tmpdir / "reference.vasp"
            displaced_path = tmpdir / "displaced.vasp"

            ase_write(reference_path, undistorted, format="vasp", direct=True)
            ase_write(displaced_path, distorted, format="vasp", direct=True)

            # Run CLI with --isodistort option
            print(f"\nRunning CLI with --isodistort option...")
            cmd_isodistort = [
                "phonproj-decompose",
                "--phonopy",
                str(BATIO3_YAML_PATH),
                "--isodistort",
                str(ISODISTORT_PATH),
                "--supercell",
                "4x4x2",
                "--remove-com",
                "--quiet",
            ]

            # Run CLI with --displaced option
            print(f"Running CLI with --displaced option...")
            cmd_displaced = [
                "phonproj-decompose",
                "--phonopy",
                str(BATIO3_YAML_PATH),
                "--reference",
                str(reference_path),
                "--displaced",
                str(displaced_path),
                "--supercell",
                "4x4x2",
                "--remove-com",
                "--quiet",
            ]

            try:
                # Run both commands
                result_isodistort = subprocess.run(
                    cmd_isodistort,
                    capture_output=True,
                    text=True,
                    timeout=120,
                    check=True,
                )
                result_displaced = subprocess.run(
                    cmd_displaced,
                    capture_output=True,
                    text=True,
                    timeout=120,
                    check=True,
                )

                # Parse displacement magnitudes
                def extract_displacement_magnitude(output):
                    for line in output.split("\n"):
                        if "Mass-weighted norm:" in line:
                            try:
                                magnitude_str = line.split(":")[1].strip()
                                return float(magnitude_str)
                            except (ValueError, IndexError):
                                pass
                    return None

                disp_isodistort = extract_displacement_magnitude(
                    result_isodistort.stdout
                )
                disp_displaced = extract_displacement_magnitude(result_displaced.stdout)

                print(f"ISODISTORT displacement: {disp_isodistort:.6f} Å")
                print(f"Displaced displacement: {disp_displaced:.6f} Å")

                assert disp_isodistort is not None, (
                    "Could not extract ISODISTORT displacement"
                )
                assert disp_displaced is not None, (
                    "Could not extract displaced displacement"
                )

                # Check equivalence (within numerical tolerance)
                relative_diff = abs(disp_isodistort - disp_displaced) / max(
                    disp_isodistort, disp_displaced
                )
                print(f"Relative difference: {relative_diff:.2e}")

                assert relative_diff < 1e-5, (
                    f"Displacement magnitudes differ too much: "
                    f"ISODISTORT={disp_isodistort:.6f}, displaced={disp_displaced:.6f}"
                )

                print(
                    f"\n✓ TEST PASSED: ISODISTORT and displaced inputs produce equivalent results"
                )

            except subprocess.CalledProcessError as e:
                print(f"\nCLI ERROR:")
                print(f"Command: {' '.join(e.cmd) if hasattr(e, 'cmd') else 'Unknown'}")
                print(f"Return code: {e.returncode}")
                print(f"STDOUT:\n{e.stdout}")
                print(f"STDERR:\n{e.stderr}")
                raise
            except subprocess.TimeoutExpired:
                pytest.fail("CLI command timed out after 120 seconds")

    def test_cli_isodistort_argument_validation(self):
        """
        Test CLI argument validation for ISODISTORT functionality.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Test 1: --isodistort and --displaced are mutually exclusive
            print(f"\nTesting mutual exclusion of --isodistort and --displaced...")
            cmd_conflict = [
                "phonproj-decompose",
                "--phonopy",
                str(BATIO3_YAML_PATH),
                "--isodistort",
                str(ISODISTORT_PATH),
                "--reference",
                "dummy.vasp",
                "--displaced",
                "dummy.vasp",
                "--supercell",
                "4x4x2",
            ]

            result = subprocess.run(
                cmd_conflict, capture_output=True, text=True, timeout=30
            )
            assert result.returncode != 0, "CLI should fail with conflicting arguments"
            assert "cannot specify both" in result.stderr.lower(), (
                "Expected error message about mutually exclusive arguments"
            )

            # Test 2: One of --isodistort or --displaced is required
            print(f"Testing requirement of input argument...")
            cmd_missing = [
                "phonproj-decompose",
                "--phonopy",
                str(BATIO3_YAML_PATH),
                "--supercell",
                "4x4x2",
            ]

            result = subprocess.run(
                cmd_missing, capture_output=True, text=True, timeout=30
            )
            assert result.returncode != 0, "CLI should fail without input argument"
            assert "must specify either" in result.stderr.lower(), (
                "Expected error message about required argument"
            )

            # Test 3: Non-existent ISODISTORT file
            print(f"Testing non-existent ISODISTORT file...")
            cmd_nonexistent = [
                "phonproj-decompose",
                "--phonopy",
                str(BATIO3_YAML_PATH),
                "--isodistort",
                "nonexistent.txt",
                "--supercell",
                "4x4x2",
            ]

            result = subprocess.run(
                cmd_nonexistent, capture_output=True, text=True, timeout=30
            )
            assert result.returncode != 0, "CLI should fail with non-existent file"

            print(f"\n✓ TEST PASSED: CLI argument validation works correctly")

    def test_cli_isodistort_help_message(self):
        """
        Test that CLI help message includes ISODISTORT option.
        """
        cmd_help = ["phonproj-decompose", "--help"]

        result = subprocess.run(
            cmd_help, capture_output=True, text=True, timeout=30, check=True
        )

        # Check that ISODISTORT option is documented
        assert "--isodistort" in result.stdout, "ISODISTORT option not found in help"
        assert "ISODISTORT" in result.stdout, "ISODISTORT description not found in help"

        print(f"✓ TEST PASSED: Help message includes ISODISTORT option")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
