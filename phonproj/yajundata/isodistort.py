import os
import sys
from pathlib import Path

# Add the project root to sys.path to enable imports from phonproj
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from phonproj.isodistort_parser import parse_isodistort_file
from ase import Atoms
from ase.io import write


def main():
    isodistort_filepath = Path("ISO_a1a2.txt")
    output_dir = Path("./isodistort_output")
    output_dir.mkdir(exist_ok=True)

    print(f"Parsing ISODISTORT file: {isodistort_filepath}")
    parsed_data = parse_isodistort_file(isodistort_filepath)

    # 1. Parent Structure
    parent_atoms = parsed_data["parent_structure"]
    parent_vasp_path = output_dir / "parent.vasp"
    write(str(parent_vasp_path), parent_atoms, format="vasp")
    print(f"Parent structure written to {parent_vasp_path}")

    # 2. Undistorted Supercell Structure
    undistorted_supercell_atoms = parsed_data["supercell_structure"]["undistorted"]
    undistorted_vasp_path = output_dir / "supercell_undistorted.vasp"
    write(str(undistorted_vasp_path), undistorted_supercell_atoms, format="vasp")
    print(f"Undistorted supercell written to {undistorted_vasp_path}")

    # 3. Distorted Supercell Structure
    distorted_supercell_atoms = parsed_data["supercell_structure"]["distorted"]
    distorted_vasp_path = output_dir / "supercell_distorted.vasp"
    write(str(distorted_vasp_path), distorted_supercell_atoms, format="vasp")
    print(f"Distorted supercell written to {distorted_vasp_path}")


if __name__ == "__main__":
    main()
