"""
Phonon Displacement Generator

A standalone tool for generating phonon mode displacements and saving supercell structures.
"""

import numpy as np
from pathlib import Path
from typing import Optional, List, Dict, Any, Union
from ase import Atoms
from ase.io import write

from phonproj.core import load_from_phonopy_files, load_yaml_file
from phonproj.core.io import _calculate_phonons_at_kpoints
from phonproj.modes import PhononModes


class PhononDisplacementGenerator:
    """
    A standalone tool for generating phonon mode displacements and saving supercell structures.

    This class provides a simple interface to:
    1. Load phonopy data from files
    2. Generate commensurate q-points for a given supercell
    3. Calculate phonon modes at those q-points
    4. Generate displacement patterns for individual modes
    5. Save supercell structures with displacements in VASP format
    """

    def __init__(self, phonopy_path: str):
        """
        Initialize the displacement generator.

        Args:
            phonopy_path: Path to phonopy_params.yaml file or directory containing phonopy files
        """
        self.phonopy_path = Path(phonopy_path)
        self.phonopy_data: Dict[str, Any] = {}
        self.phonon_modes: Optional[PhononModes] = None
        self._load_phonopy_data()

    def _load_phonopy_data(self):
        """Load phonopy data from specified path."""
        if self.phonopy_path.is_dir():
            self.phonopy_data = load_from_phonopy_files(self.phonopy_path)
        else:
            self.phonopy_data = load_yaml_file(self.phonopy_path)

    def calculate_modes(self, supercell_matrix: np.ndarray) -> PhononModes:
        """
        Calculate phonon modes for commensurate q-points of the given supercell.

        Args:
            supercell_matrix: 3x3 supercell transformation matrix

        Returns:
            PhononModes object with calculated modes
        """
        # Generate commensurate q-points for the supercell
        n1, n2, n3 = (
            int(supercell_matrix[0, 0]),
            int(supercell_matrix[1, 1]),
            int(supercell_matrix[2, 2]),
        )

        qpoints = []
        for i in range(n1):
            for j in range(n2):
                for k in range(n3):
                    qpoints.append([i / n1, j / n2, k / n3])
        qpoints = np.array(qpoints)

        # Get the phonopy object and calculate modes
        phonopy = self.phonopy_data["phonopy"]
        primitive_cell = self.phonopy_data["primitive_cell"]

        # Calculate phonon modes at commensurate q-points
        frequencies, eigenvectors = _calculate_phonons_at_kpoints(phonopy, qpoints)

        # Create PhononModes object
        self.phonon_modes = PhononModes(
            primitive_cell=primitive_cell,
            qpoints=qpoints,
            frequencies=frequencies,
            eigenvectors=eigenvectors,
            atomic_masses=None,  # Will be inferred from primitive_cell
            gauge="R",
        )

        return self.phonon_modes

    def get_commensurate_qpoints(self, supercell_matrix: np.ndarray) -> List[int]:
        """
        Get indices of commensurate q-points for the given supercell.

        Args:
            supercell_matrix: 3x3 supercell transformation matrix

        Returns:
            List of q-point indices
        """
        # Always calculate modes for the current supercell to ensure correctness
        self.calculate_modes(supercell_matrix)

        assert self.phonon_modes is not None  # Type assertion
        result = self.phonon_modes.get_commensurate_qpoints(supercell_matrix)
        # Handle both list and dict return types
        if isinstance(result, dict):
            return result["matched_indices"]
        return result

    def generate_displacement(
        self,
        q_idx: int,
        mode_idx: int,
        supercell_matrix: np.ndarray,
        amplitude: float = 0.1,
    ) -> np.ndarray:
        """
        Generate displacement pattern for a specific phonon mode.

        Args:
            q_idx: Q-point index
            mode_idx: Mode index
            supercell_matrix: 3x3 supercell transformation matrix
            amplitude: Displacement amplitude

        Returns:
            Displacement vector array
        """
        if self.phonon_modes is None:
            self.calculate_modes(supercell_matrix)

        assert self.phonon_modes is not None  # Type assertion
        return self.phonon_modes.generate_mode_displacement(
            q_idx, mode_idx, supercell_matrix, amplitude=amplitude
        )

    def generate_supercell_structure(
        self,
        q_idx: int,
        mode_idx: int,
        supercell_matrix: np.ndarray,
        amplitude: float = 0.1,
    ) -> Atoms:
        """
        Generate supercell structure with displacement for a specific phonon mode.

        Args:
            q_idx: Q-point index
            mode_idx: Mode index
            supercell_matrix: 3x3 supercell transformation matrix
            amplitude: Displacement amplitude

        Returns:
            ASE Atoms object with displaced structure
        """
        if self.phonon_modes is None:
            self.calculate_modes(supercell_matrix)

        assert self.phonon_modes is not None  # Type assertion

        # Generate displacement
        displacement = self.generate_displacement(
            q_idx, mode_idx, supercell_matrix, amplitude
        )

        # Generate base supercell structure
        supercell_structure = self.phonon_modes.generate_supercell(supercell_matrix)

        # Apply displacement to supercell
        supercell_structure.set_positions(
            supercell_structure.get_positions() + displacement
        )

        return supercell_structure

    def print_displacements(
        self,
        supercell_matrix: np.ndarray,
        amplitude: float = 0.1,
        max_atoms_per_mode: int = 5,
    ):
        """
        Print all supercell displacements for commensurate q-points.

        Args:
            supercell_matrix: 3x3 supercell transformation matrix
            amplitude: Displacement amplitude
            max_atoms_per_mode: Maximum number of atoms to show per mode
        """
        if self.phonon_modes is None:
            self.calculate_modes(supercell_matrix)

        assert self.phonon_modes is not None  # Type assertion

        print(f"\n=== Supercell Displacements (amplitude = {amplitude}) ===")

        # Get all commensurate q-points
        commensurate_qpoints = self.get_commensurate_qpoints(supercell_matrix)

        if not commensurate_qpoints:
            print("No commensurate q-points found for the given supercell matrix.")
            return

        print(f"Found {len(commensurate_qpoints)} commensurate q-points:")

        for q_idx in commensurate_qpoints:
            q_idx_int = int(q_idx)
            qpoint = self.phonon_modes.qpoints[q_idx_int]
            print(
                f"\nQ-point {q_idx_int}: [{qpoint[0]:.3f}, {qpoint[1]:.3f}, {qpoint[2]:.3f}]"
            )
            print("-" * 50)

            # Generate displacements for all modes at this q-point
            for mode_idx in range(self.phonon_modes.n_modes):
                try:
                    displacement = self.generate_displacement(
                        q_idx_int, mode_idx, supercell_matrix, amplitude=amplitude
                    )

                    # Print displacement info with limited precision
                    freq = self.phonon_modes.frequencies[q_idx_int, mode_idx]
                    print(f"Mode {mode_idx:2d} (freq = {freq:8.2f} cm⁻¹):")

                    # Print a few representative atoms
                    n_atoms_to_show = min(max_atoms_per_mode, len(displacement))
                    for atom_idx in range(n_atoms_to_show):
                        disp = displacement[atom_idx]
                        if np.iscomplexobj(disp):
                            # Print complex displacement
                            print(
                                f"  Atom {atom_idx:2d}: ({disp.real:8.4f}, {disp.imag:8.4f}i) Å"
                            )
                        else:
                            # Print real displacement
                            print(
                                f"  Atom {atom_idx:2d}: ({disp[0]:8.4f}, {disp[1]:8.4f}, {disp[2]:8.4f}) Å"
                            )

                    if len(displacement) > n_atoms_to_show:
                        print(
                            f"  ... and {len(displacement) - n_atoms_to_show} more atoms"
                        )

                except Exception as e:
                    print(f"  Mode {mode_idx:2d}: Error generating displacement - {e}")

    def save_structure(
        self,
        q_idx: int,
        mode_idx: int,
        supercell_matrix: np.ndarray,
        amplitude: float = 0.1,
        output_dir: str = ".",
    ) -> str:
        """
        Save a single displaced supercell structure to VASP format.

        Args:
            q_idx: Q-point index
            mode_idx: Mode index
            supercell_matrix: 3x3 supercell transformation matrix
            amplitude: Displacement amplitude
            output_dir: Directory to save VASP file

        Returns:
            Path to the saved VASP file
        """
        if self.phonon_modes is None:
            self.calculate_modes(supercell_matrix)

        assert self.phonon_modes is not None  # Type assertion

        # Generate displacement
        displacement = self.generate_displacement(
            q_idx, mode_idx, supercell_matrix, amplitude
        )

        # Generate base supercell structure
        supercell_structure = self.phonon_modes.generate_supercell(supercell_matrix)

        # Apply displacement to supercell
        supercell_structure.set_positions(
            supercell_structure.get_positions() + displacement
        )

        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Generate filename
        frequency = self.phonon_modes.frequencies[q_idx, mode_idx]
        filename = f"mode_q{q_idx}_m{mode_idx}_freq_{frequency:.2f}THz.vasp"
        filepath = output_path / filename

        # Write VASP file
        write(filepath, supercell_structure, format="vasp")

        return str(filepath)

    def save_all_structures(
        self, supercell_matrix: np.ndarray, output_dir: str, amplitude: float = 0.1
    ) -> Dict[str, Any]:
        """
        Save all supercell structures with displacements to directory in VASP format.

        Args:
            supercell_matrix: 3x3 supercell transformation matrix
            output_dir: Directory to save VASP files
            amplitude: Displacement amplitude

        Returns:
            Dictionary with summary information
        """
        if self.phonon_modes is None:
            self.calculate_modes(supercell_matrix)

        assert self.phonon_modes is not None  # Type assertion

        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        print(f"\n=== Saving Supercell Structures to {output_dir} ===")

        # Get all commensurate q-points
        commensurate_qpoints = self.get_commensurate_qpoints(supercell_matrix)

        if not commensurate_qpoints:
            print("No commensurate q-points found for the given supercell matrix.")
            return {"saved_files": [], "total_saved": 0}

        print(f"Found {len(commensurate_qpoints)} commensurate q-points")

        saved_files = []

        # Save structures for each commensurate q-point and mode
        for q_idx in commensurate_qpoints:
            q_idx_int = int(q_idx)
            qpoint = self.phonon_modes.qpoints[q_idx_int]
            q_str = f"q{q_idx_int}_{''.join(f'{c:.2f}'.replace('.', 'p').replace('-', 'm') for c in qpoint)}"

            for mode_idx in range(self.phonon_modes.n_modes):
                try:
                    # Generate supercell structure with displacement
                    supercell_structure = self.generate_supercell_structure(
                        q_idx_int, mode_idx, supercell_matrix, amplitude=amplitude
                    )

                    # Create filename
                    freq = self.phonon_modes.frequencies[q_idx_int, mode_idx]
                    filename = f"{q_str}_mode{mode_idx:02d}_freq{freq:6.1f}.vasp"
                    filepath = output_path / filename

                    # Save in VASP format
                    write(filepath, supercell_structure, format="vasp")
                    saved_files.append(str(filepath))

                    print(f"  Saved: {filename}")

                except Exception as e:
                    print(f"  Error saving q{q_idx_int}_mode{mode_idx}: {e}")

        total_saved = len(saved_files)
        print(f"\n✅ Saved {total_saved} supercell structures to {output_dir}")

        return {
            "saved_files": saved_files,
            "total_saved": total_saved,
            "output_dir": str(output_path),
            "amplitude": amplitude,
            "supercell_matrix": supercell_matrix.tolist(),
        }
