"""
Test the updated _run_irreps_analysis implementation.

Purpose:
    Verify that the refactored _IrRepsLocal class now produces proper symmetry labels.

How to run:
    uv run python agent_files/debug_irreps/test_irreps_simple.py

Expected behavior:
    Should print actual irrep labels (Ag, B1u, etc.) instead of None.
"""

import numpy as np
import sys

# Simple test - create fake phonon data
from phonopy.structure.atoms import PhonopyAtoms
from phonopy.phonon.irreps import IrReps, IrRepLabels
from phonopy.structure.symmetry import Symmetry
from phonopy.phonon.character_table import character_table
from phonopy.phonon.degeneracy import degenerate_sets as get_degenerate_sets
from phonopy.structure.cells import is_primitive_cell

print("=" * 80)
print("Testing _IrRepsLocal Implementation")
print("=" * 80)

# Create a simple cubic structure for testing - single atom primitive cell
lattice = np.eye(3) * 5.0
positions = np.array([[0.0, 0.0, 0.0]])
numbers = [1]

phonopy_atoms = PhonopyAtoms(cell=lattice, scaled_positions=positions, numbers=numbers)

# Create dummy frequencies and eigenvectors for Gamma point
n_modes = 3  # 3 modes for 1 atom
qpoint = np.array([0.0, 0.0, 0.0])
freqs = np.array([1.0, 1.0, 1.0])
eigvecs = np.random.random((n_modes, n_modes)) + 1j * np.random.random(
    (n_modes, n_modes)
)

print(f"\nTest structure: 1 atom (primitive)")
print(f"Q-point: {qpoint}")
print(f"Frequencies: {freqs}")


# Create the local IrReps class
class _IrRepsLocal(IrReps, IrRepLabels):
    """Local IrReps implementation without DynamicalMatrix dependency."""

    def __init__(self, primitive, qpoint, freqs, eigvecs, symprec, deg_tol):
        self._is_little_cogroup = False
        self._log_level = 0
        self._qpoint = np.array(qpoint)
        self._degeneracy_tolerance = deg_tol
        self._symprec = symprec
        self._primitive = primitive
        self._freqs, self._eig_vecs = freqs, eigvecs
        self._character_table = None
        self._verbose = 2  # Enable maximum verbose output

    def run(self):
        """Run the irreps analysis following IrRepsEigen pattern."""
        # Get symmetry dataset
        symmetry = Symmetry(self._primitive, symprec=self._symprec)
        self._symmetry_dataset = symmetry.get_dataset()

        # Check if primitive
        if not is_primitive_cell(self._symmetry_dataset["rotations"]):
            raise RuntimeError("Non-primitive cell is used.")

        # Get rotations at q
        self._rotations_at_q, self._translations_at_q = self._get_rotations_at_q()
        self._g = len(self._rotations_at_q)

        # Get point group info
        self._pointgroup_symbol = self._symmetry_dataset.pointgroup
        print(f"\nPoint group: {self._pointgroup_symbol}")

        # Get transformation matrix
        self._transformation_matrix, self._conventional_rotations = (
            self._get_conventional_rotations()
        )

        # Calculate irreps
        self._ground_matrices = self._get_ground_matrix()
        self._degenerate_sets = self._get_degenerate_sets()
        print(f"Degenerate sets: {self._degenerate_sets}")

        self._irreps = self._get_irreps()
        print(f"Number of irreps: {len(self._irreps)}")

        self._characters, self._irrep_dims = self._get_characters()
        print(
            f"Characters shape: {self._characters.shape if hasattr(self._characters, 'shape') else len(self._characters)}"
        )
        print(f"Irrep dims: {self._irrep_dims}")

        # Get irrep labels
        self._ir_labels = None

        if (
            self._pointgroup_symbol in character_table.keys()
            and character_table[self._pointgroup_symbol] is not None
        ):
            print(f"\nCharacter table available for {self._pointgroup_symbol}")

            self._rotation_symbols, character_table_of_ptg = self._get_rotation_symbols(
                self._pointgroup_symbol
            )
            self._character_table = character_table_of_ptg

            print(f"Rotation symbols: {self._rotation_symbols}")
            print(f"Character table keys: {list(character_table_of_ptg.keys())}")

            if (abs(self._qpoint) < self._symprec).all() and self._rotation_symbols:
                print(f"\nCalling _get_irrep_labels...")
                self._ir_labels = self._get_irrep_labels(character_table_of_ptg)
                print(f"IR labels: {self._ir_labels}")

        else:
            self._rotation_symbols = None
            print(f"No character table for {self._pointgroup_symbol}")

        return True

    def _get_degenerate_sets(self):
        """Get degenerate sets - needed to override parent method."""
        return get_degenerate_sets(self._freqs, cutoff=self._degeneracy_tolerance)


# Test the implementation
try:
    irreps = _IrRepsLocal(
        phonopy_atoms, qpoint, freqs, eigvecs, symprec=1e-5, deg_tol=1e-4
    )
    success = irreps.run()

    print("\n" + "=" * 80)
    print("Results")
    print("=" * 80)

    if success:
        print(f"✓ Analysis completed successfully")
        print(f"\nIR labels: {irreps._ir_labels}")
        print(f"Number of labels: {len(irreps._ir_labels) if irreps._ir_labels else 0}")

        if irreps._ir_labels:
            print(f"\nLabels by degenerate set:")
            for i, label in enumerate(irreps._ir_labels):
                print(f"  Set {i}: {label}")
        else:
            print(f"\n✗ IR labels are None")
    else:
        print(f"✗ Analysis failed")

except Exception as e:
    print(f"\n✗ Exception occurred: {e}")
    import traceback

    traceback.print_exc()
