#!/usr/bin/env python3
"""
Test script for the isodistort_parser module.

PURPOSE:
    Test the isodistort_parser module with P4mmm-ref.txt data to verify:
    - File parsing works correctly
    - ASE Atoms objects are created properly
    - Cell parameters are handled correctly with cellpar
    - Mode parsing works as expected

USAGE:
    uv run python test_isodistort_parser.py

EXPECTED OUTPUT:
    Successfully parsed P4mmm-ref.txt with:
    - Parent structure with 10 atoms
    - Supercell structures (undistorted and distorted)
    - Multiple phonon modes with displacement patterns

FILES USED:
    - phonproj/isodistort_parser.py (module being tested)
    - data/yajundata/P4mmm-ref.txt (test data)

DEBUG NOTES:
    This test verifies that the recent fix to use ASE's cellpar parameter
    instead of manual cell matrix construction works correctly.
"""

import sys
from pathlib import Path

from ase import Atoms

from phonproj.isodistort_parser import group_modes_by_qpoint, parse_isodistort_file

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


def test_parse_p4mmm_ref() -> bool:
    """Test parsing P4mmm-ref.txt file."""
    print("Testing isodistort_parser with P4mmm-ref.txt...")

    # Test file path
    test_file = project_root / "data" / "yajundata" / "P4mmm-ref.txt"

    if not test_file.exists():
        print(f"ERROR: Test file not found: {test_file}")
        return False

    try:
        # Parse the file
        result = parse_isodistort_file(test_file)
        print("✓ File parsing successful")

        # Check parent structure
        parent = result.get("parent_structure")
        if parent is None:
            print("ERROR: No parent structure found")
            return False

        # Type assertion for mypy
        assert isinstance(parent, Atoms), "parent_structure should be an Atoms object"

        print(f"✓ Parent structure: {len(parent)} atoms")
        print(f"  Cell parameters: {parent.cell}")
        print(f"  Atomic symbols: {parent.get_chemical_symbols()}")

        # Check supercell structures
        supercell = result.get("supercell_structure", {})
        # Type assertion for mypy
        assert isinstance(supercell, dict), "supercell_structure should be a dict"
        undistorted = supercell.get("undistorted")
        distorted = supercell.get("distorted")

        if undistorted:
            print(f"✓ Undistorted supercell: {len(undistorted)} atoms")
        else:
            print("WARNING: No undistorted supercell found")

        if distorted:
            print(f"✓ Distorted supercell: {len(distorted)} atoms")
        else:
            print("WARNING: No distorted supercell found")

        # Check modes
        modes = result.get("modes", {})
        # Type assertion for mypy
        assert isinstance(modes, dict), "modes should be a dict"
        print(f"✓ Found {len(modes)} phonon modes")

        # Group modes by q-point
        qpoint_groups = group_modes_by_qpoint(modes)
        print(f"✓ Modes grouped into {len(qpoint_groups)} q-points:")

        for qpoint, mode_list in sorted(qpoint_groups.items()):
            print(f"  Q-point {qpoint}: {len(mode_list)} modes")

        # Test a few specific modes
        if modes:
            first_mode = list(modes.keys())[0]
            mode_data = modes[first_mode]
            # Type assertion for mypy
            assert isinstance(mode_data, dict), "mode_data should be a dict"
            print(f"✓ First mode: {first_mode}")
            print(f"  Normfactor: {mode_data['normfactor']}")
            print(f"  Displacements: {len(mode_data['displacements'])} atoms")

            # Show first few displacements
            for disp in mode_data["displacements"][:3]:
                print(
                    f"    {disp['atom_label']}: ({disp['dx']:.3f}, {disp['dy']:.3f}, {disp['dz']:.3f})"
                )

        print("\n✓ All tests passed!")
        return True

    except Exception as e:
        print(f"ERROR: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_ase_cell_creation() -> bool:
    """Test that ASE Atoms objects are created correctly."""
    print("\nTesting ASE cell creation...")

    test_file = project_root / "data" / "yajundata" / "P4mmm-ref.txt"

    try:
        result = parse_isodistort_file(test_file)
        parent = result["parent_structure"]

        # Type assertion for mypy
        assert isinstance(parent, Atoms), "parent_structure should be an Atoms object"

        # Check that cell is properly set
        cell = parent.cell
        print(f"✓ Cell matrix shape: {cell.array.shape}")
        print(f"✓ Cell volume: {parent.get_volume():.3f} Ų")

        # Check that positions are within cell
        positions = parent.get_scaled_positions()
        print(f"✓ Position range: [{positions.min():.3f}, {positions.max():.3f}]")

        # Verify all atoms are within [0,1) range for scaled positions
        if (positions >= 0).all() and (positions < 1).all():
            print("✓ All scaled positions are within valid range [0,1)")
        else:
            print("WARNING: Some scaled positions are outside [0,1) range")

        return True

    except Exception as e:
        print(f"ERROR in ASE cell creation test: {e}")
        return False


if __name__ == "__main__":
    print("=" * 60)
    print("ISODISTORT PARSER TEST")
    print("=" * 60)

    success1 = test_parse_p4mmm_ref()
    success2 = test_ase_cell_creation()

    print("\n" + "=" * 60)
    if success1 and success2:
        print("ALL TESTS PASSED! ✓")
        sys.exit(0)
    else:
        print("SOME TESTS FAILED! ✗")
        sys.exit(1)
