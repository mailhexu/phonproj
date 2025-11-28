"""
Tests for Phonon Displacement Generator CLI
"""

import pytest
import subprocess
import tempfile
import numpy as np
from pathlib import Path


class TestDisplacementCLI:
    """Test the displacement generator CLI."""

    @pytest.fixture
    def batio3_data(self):
        """Get BaTiO3 phonopy data path."""
        return (
            Path(__file__).parent.parent.parent / "data" / "BaTiO3_phonopy_params.yaml"
        )

    @pytest.fixture
    def cli_command(self):
        """Get the CLI command."""
        return ["uv", "run", "phonproj-displacement"]

    def test_cli_help(self, cli_command):
        """Test CLI help output."""
        result = subprocess.run(
            cli_command + ["--help"], capture_output=True, text=True
        )

        assert result.returncode == 0
        assert "Generate phonon mode displacements" in result.stdout
        assert "--phonopy" in result.stdout
        assert "--supercell" in result.stdout
        assert "--print-displacements" in result.stdout
        assert "--save-dir" in result.stdout

    def test_cli_missing_action(self, cli_command, batio3_data):
        """Test CLI error when no action specified."""
        result = subprocess.run(
            cli_command + ["-p", str(batio3_data), "-s", "2x2x2"],
            capture_output=True,
            text=True,
        )

        assert result.returncode != 0
        assert (
            "Must specify either --print-displacements or --save-dir" in result.stderr
        )

    def test_cli_print_displacements(self, cli_command, batio3_data):
        """Test CLI displacement printing."""
        result = subprocess.run(
            cli_command
            + [
                "-p",
                str(batio3_data),
                "-s",
                "2x2x2",
                "--print-displacements",
                "--amplitude",
                "0.05",
                "--max-atoms",
                "2",
            ],
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0
        assert "Supercell Displacements" in result.stdout
        assert "Found 8 commensurate q-points" in result.stdout
        assert "Q-point 0:" in result.stdout
        assert "Mode" in result.stdout
        assert "freq" in result.stdout
        assert "completed successfully" in result.stdout

    def test_cli_save_structures(self, cli_command, batio3_data):
        """Test CLI structure saving."""
        with tempfile.TemporaryDirectory() as temp_dir:
            result = subprocess.run(
                cli_command
                + [
                    "-p",
                    str(batio3_data),
                    "-s",
                    "2x2x2",
                    "--save-dir",
                    temp_dir,
                    "--amplitude",
                    "0.05",
                ],
                capture_output=True,
                text=True,
            )

            assert result.returncode == 0
            assert "Saving Supercell Structures" in result.stdout
            assert "Found 8 commensurate q-points" in result.stdout
            assert "completed successfully" in result.stdout

            # Check that files were created
            output_path = Path(temp_dir)
            vasp_files = list(output_path.glob("*.vasp"))
            assert len(vasp_files) == 120  # 8 q-points × 15 modes

    def test_cli_both_actions(self, cli_command, batio3_data):
        """Test CLI with both print and save actions."""
        with tempfile.TemporaryDirectory() as temp_dir:
            result = subprocess.run(
                cli_command
                + [
                    "-p",
                    str(batio3_data),
                    "-s",
                    "2x2x2",
                    "--print-displacements",
                    "--save-dir",
                    temp_dir,
                    "--amplitude",
                    "0.1",
                ],
                capture_output=True,
                text=True,
            )

            assert result.returncode == 0
            assert "Supercell Displacements" in result.stdout
            assert "Saving Supercell Structures" in result.stdout
            assert "completed successfully" in result.stdout

            # Check files were created
            output_path = Path(temp_dir)
            vasp_files = list(output_path.glob("*.vasp"))
            assert len(vasp_files) == 120

    def test_cli_different_supercell_sizes(self, cli_command, batio3_data):
        """Test CLI with different supercell sizes."""
        # Test 4x1x1 supercell
        result = subprocess.run(
            cli_command
            + [
                "-p",
                str(batio3_data),
                "-s",
                "4x1x1",
                "--print-displacements",
                "--quiet",
            ],
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0
        assert "Found 4 commensurate q-points" in result.stdout

        # Test 3x3x3 supercell
        result = subprocess.run(
            cli_command
            + [
                "-p",
                str(batio3_data),
                "-s",
                "3x3x3",
                "--print-displacements",
                "--quiet",
            ],
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0
        assert "Found 27 commensurate q-points" in result.stdout

    def test_cli_invalid_supercell(self, cli_command, batio3_data):
        """Test CLI with invalid supercell format."""
        result = subprocess.run(
            cli_command
            + ["-p", str(batio3_data), "-s", "invalid", "--print-displacements"],
            capture_output=True,
            text=True,
        )

        assert result.returncode != 0
        assert "Supercell must have 3 dimensions" in result.stderr

    def test_cli_missing_phonopy(self, cli_command):
        """Test CLI with missing phonopy file."""
        result = subprocess.run(
            cli_command
            + ["-p", "nonexistent.yaml", "-s", "2x2x2", "--print-displacements"],
            capture_output=True,
            text=True,
        )

        assert result.returncode != 0
        assert "Error:" in result.stderr

    def test_cli_quiet_mode(self, cli_command, batio3_data):
        """Test CLI quiet mode."""
        result = subprocess.run(
            cli_command
            + [
                "-p",
                str(batio3_data),
                "-s",
                "2x2x2",
                "--print-displacements",
                "--quiet",
            ],
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0
        # Should not have verbose loading messages
        assert "Loading phonopy data" not in result.stdout
        assert "Supercell:" not in result.stdout
        # But should still have the main output
        assert "Supercell Displacements" in result.stdout
