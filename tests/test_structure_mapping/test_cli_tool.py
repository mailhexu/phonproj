"""
Test script for map_structure CLI tool.

This script tests various CLI options to ensure the tool works correctly.
"""

import subprocess
import os
import tempfile
from pathlib import Path


def run_cli(args, expected_success=True):
    """Run CLI command and return result."""
    cmd = ["uv", "run", "python", "-m", "phonproj.map_structure"] + args
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=".")

    if expected_success:
        assert result.returncode == 0, (
            f"CLI failed with args: {args}\nSTDOUT: {result.stdout}\nSTDERR: {result.stderr}"
        )
    else:
        assert result.returncode != 0, f"CLI should have failed with args: {args}"

    return result.stdout, result.stderr


def test_cli_help():
    """Test CLI help functionality."""
    print("Testing CLI help...")
    stdout, stderr = run_cli(["--help"])

    assert "Enhanced structure mapping tool" in stdout
    assert "reference" in stdout
    assert "target" in stdout
    assert "--species-map" in stdout
    print("✓ Help test passed")


def test_cli_info_only():
    """Test structure info only functionality."""
    print("Testing info-only option...")

    # Use real test data if available
    ref_file = "data/yajundata/ref.vasp"
    target_file = "data/yajundata/SM2.vasp"

    if not os.path.exists(ref_file) or not os.path.exists(target_file):
        print("⚠ Skipping info test - test data not available")
        return

    stdout, stderr = run_cli([ref_file, target_file, "--info-only"])

    assert "Reference Structure:" in stdout
    assert "Target Structure:" in stdout
    assert "Atoms: 160" in stdout
    print("✓ Info-only test passed")


def test_cli_basic_mapping():
    """Test basic mapping functionality."""
    print("Testing basic mapping...")

    ref_file = "data/yajundata/ref.vasp"
    target_file = "data/yajundata/SM2.vasp"

    if not os.path.exists(ref_file) or not os.path.exists(target_file):
        print("⚠ Skipping basic mapping test - test data not available")
        return

    # Test with quiet mode to reduce output
    stdout, stderr = run_cli([ref_file, target_file, "--method", "distance", "--quiet"])

    assert "Mapping Results:" in stdout
    assert "Total atoms mapped: 160" in stdout
    print("✓ Basic mapping test passed")


def test_cli_enhanced_mapping():
    """Test enhanced mapping functionality."""
    print("Testing enhanced mapping...")

    ref_file = "data/yajundata/ref.vasp"
    target_file = "data/yajundata/SM2.vasp"

    if not os.path.exists(ref_file) or not os.path.exists(target_file):
        print("⚠ Skipping enhanced mapping test - test data not available")
        return

    # Test with output file
    with tempfile.TemporaryDirectory() as temp_dir:
        output_file = os.path.join(temp_dir, "test_output.txt")

        stdout, stderr = run_cli(
            [
                ref_file,
                target_file,
                "--method",
                "enhanced",
                "--output",
                output_file,
                "--quiet",
            ]
        )

        assert os.path.exists(output_file), "Output file was not created"

        with open(output_file, "r") as f:
            content = f.read()
            assert "ENHANCED ATOM MAPPING ANALYSIS REPORT" in content
            assert "MAPPING SUMMARY" in content
            assert "DETAILED MAPPING TABLE" in content

    print("✓ Enhanced mapping test passed")


def test_cli_error_handling():
    """Test CLI error handling."""
    print("Testing error handling...")

    # Test with non-existent file
    stdout, stderr = run_cli(
        ["nonexistent.vasp", "also_nonexistent.vasp"], expected_success=False
    )

    assert "Structure file not found" in stderr
    print("✓ Error handling test passed")


def test_cli_species_mapping():
    """Test species mapping functionality."""
    print("Testing species mapping...")

    ref_file = "data/yajundata/ref.vasp"
    target_file = "data/yajundata/SM2.vasp"

    if not os.path.exists(ref_file) or not os.path.exists(target_file):
        print("⚠ Skipping species mapping test - test data not available")
        return

    # Test with species mapping (without quiet to see the message)
    stdout, stderr = run_cli([ref_file, target_file, "--species-map", "Pb:Sr,Ti:Zr"])

    assert "Using species mapping:" in stdout
    print("✓ Species mapping test passed")


def main():
    """Run all CLI tests."""
    print("Testing map_structure CLI tool...")
    print("=" * 50)

    try:
        test_cli_help()
        test_cli_info_only()
        test_cli_basic_mapping()
        test_cli_enhanced_mapping()
        test_cli_error_handling()
        test_cli_species_mapping()

        print("=" * 50)
        print("✅ All CLI tests passed!")

    except AssertionError as e:
        print(f"❌ Test failed: {e}")
        return 1
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
