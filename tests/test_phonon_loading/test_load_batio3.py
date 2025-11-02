"""Test loading BaTiO3 phonon data from YAML file."""

# import numpy as np
from pathlib import Path

from phonproj.core import load_yaml_file, create_phonopy_object


def test_load_batio3_yaml():
    """Test loading BaTiO3 from YAML file."""
    yaml_path = "/Users/hexu/projects/phonproj/data/BaTiO3_phonopy_params.yaml"

    # Load from YAML
    data = load_yaml_file(yaml_path)

    # Verify structure
    assert "phonopy" in data
    assert "primitive_cell" in data
    assert "unitcell" in data
    assert "supercell" in data

    # Verify phonopy object
    phonopy = data["phonopy"]
    assert phonopy is not None

    # Verify primitive cell
    primitive = data["primitive_cell"]
    assert len(primitive) > 0  # Should have atoms

    print(f"✓ BaTiO3 YAML loaded successfully")
    print(f"  Primitive cell: {len(primitive)} atoms")
    # Frequencies are not available until calculation is run
    # print(f"  Frequencies shape: {phonopy.frequencies.shape}")


def test_create_phonopy_object():
    """Test creating phonopy object directly."""
    yaml_path = "/Users/hexu/projects/phonproj/data/BaTiO3_phonopy_params.yaml"

    phonopy = create_phonopy_object(yaml_path)

    assert phonopy is not None
    # Frequencies and eigenvectors are not available until calculation is run
    # assert hasattr(phonopy, 'frequencies')
    # assert hasattr(phonopy, 'eigenvectors')
    print(f"✓ Phonopy object created successfully")


if __name__ == "__main__":
    test_load_batio3_yaml()
    test_create_phonopy_object()
    print("✅ All BaTiO3 loading tests passed!")
