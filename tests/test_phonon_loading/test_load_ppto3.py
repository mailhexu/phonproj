"""Test loading PbTiO3 phonon data from directory."""

# import numpy as np
from pathlib import Path

from phonproj.core import load_from_phonopy_files


def test_load_ppto3_directory():
    """Test loading PbTiO3 from directory with FORCE_SETS."""
    directory = Path("data/yajundata/0.02-P4mmm-PTO")

    # Load from directory
    data = load_from_phonopy_files(directory)

    # Verify structure
    assert "phonopy" in data
    assert "primitive_cell" in data
    assert "unitcell" in data
    assert "supercell" in data
    assert "directory" in data

    # Verify phonopy object
    phonopy = data["phonopy"]
    assert phonopy is not None

    # Verify structures
    primitive = data["primitive_cell"]
    unitcell = data["unitcell"]
    supercell = data["supercell"]

    assert len(primitive) > 0
    assert len(unitcell) > 0
    assert len(supercell) > 0

    # Supercell should have more atoms than primitive
    assert len(supercell) > len(primitive)

    print(f"✓ PbTiO3 directory loaded successfully")
    print(f"  Primitive cell: {len(primitive)} atoms")
    print(f"  Unit cell: {len(unitcell)} atoms")
    print(f"  Supercell: {len(supercell)} atoms")
    # Frequencies are not available until calculation is run
    # print(f"  Frequencies shape: {phonopy.frequencies.shape}")


def test_directory_path_stored():
    """Test that directory path is stored in result."""
    directory = Path("data/yajundata/0.02-P4mmm-PTO")

    data = load_from_phonopy_files(directory)

    assert "directory" in data
    assert data["directory"] == str(directory)

    print(f"✓ Directory path stored correctly: {data['directory']}")


if __name__ == "__main__":
    test_load_ppto3_directory()
    test_directory_path_stored()
    print("\n✅ All PbTiO3 loading tests passed!")
