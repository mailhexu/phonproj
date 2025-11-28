"""
Debug script to test irreps analysis.

Purpose:
    Test the _run_irreps_analysis method to understand why labels are None.

How to run:
    uv run python agent_files/debug_irreps/test_irreps_debug.py

Expected behavior:
    Should print debug information about the irreps calculation process.
"""

import numpy as np
from phonproj.modes import PhononModes

# Load phonon data
phonon_yaml = "yajundata/0.02-P4mmm-PTO/phonopy.yaml"
# Just analyze Gamma point
qpoints = np.array([[0.0, 0.0, 0.0]])
modes = PhononModes.from_phonopy_yaml(phonon_yaml, qpoints)

print("=" * 80)
print("Testing IrReps Analysis Debug")
print("=" * 80)

# Get Gamma point data
q_index = 0
qpoint = modes.qpoints[q_index]
freqs = modes.frequencies[q_index]
eigvecs = modes.eigenvectors[q_index]

print(f"\nQ-point: {qpoint}")
print(f"Number of modes: {len(freqs)}")
print(f"Frequencies (first 5): {freqs[:5]}")

# Now let's manually run the irreps analysis with debugging
from phonopy.structure.atoms import PhonopyAtoms
from phonopy.phonon.irreps import IrReps, IrRepLabels
from phonopy.structure.symmetry import Symmetry
from phonopy.phonon.character_table import character_table
from phonopy.phonon.degeneracy import degenerate_sets as get_degenerate_sets
from phonopy.structure.cells import is_primitive_cell

# Convert ASE Atoms to PhonopyAtoms
cell = modes.primitive_cell.get_cell()
scaled_positions = modes.primitive_cell.get_scaled_positions()
numbers = modes.primitive_cell.get_atomic_numbers()

phonopy_atoms = PhonopyAtoms(
    cell=cell, scaled_positions=scaled_positions, numbers=numbers
)

print("\n" + "=" * 80)
print("Running IrReps Analysis with Debug Output")
print("=" * 80)

symprec = 1e-5
deg_tol = 1e-4


# Create a debug version of the local class
class _IrRepsDebug(IrReps, IrRepLabels):
    """Debug version with print statements."""

    def __init__(self, primitive, qpoint, freqs, eigvecs, symprec, deg_tol):
        self._primitive = primitive
        self._qpoint = np.array(qpoint)
        self._freqs = freqs
        self._eig_vecs = eigvecs
        self._symprec = symprec
        self._degeneracy_tolerance = deg_tol
        self._is_little_cogroup = False
        self._log_level = 0
        self._verbose = 2  # Enable verbose output
        self._character_table = None

    def run_analysis(self):
        """Run the irreps analysis with debug output."""
        # Get symmetry dataset
        symmetry = Symmetry(self._primitive, symprec=self._symprec)
        self._symmetry_dataset = symmetry.get_dataset()

        # Check if primitive
        if not is_primitive_cell(self._symmetry_dataset["rotations"]):
            print("ERROR: Non-primitive cell detected!")
            return False

        print(f"✓ Primitive cell check passed")

        # Get point group info
        self._pointgroup_symbol = self._symmetry_dataset.pointgroup
        print(f"✓ Point group: {self._pointgroup_symbol}")

        # Get rotations at q
        self._rotations_at_q, self._translations_at_q = self._get_rotations_at_q()
        self._g = len(self._rotations_at_q)
        print(f"✓ Number of symmetry operations at q: {self._g}")

        # Get transformation matrix
        self._transformation_matrix, self._conventional_rotations = (
            self._get_conventional_rotations()
        )
        print(
            f"✓ Number of conventional rotations: {len(self._conventional_rotations)}"
        )

        # Calculate irreps
        self._ground_matrices = self._get_ground_matrix()
        print(f"✓ Ground matrices calculated: shape = {self._ground_matrices.shape}")

        self._degenerate_sets = get_degenerate_sets(
            self._freqs, cutoff=self._degeneracy_tolerance
        )
        print(f"✓ Degenerate sets: {len(self._degenerate_sets)} sets")
        print(f"  Sets: {self._degenerate_sets[:10]}")  # First 10 sets

        self._irreps = self._get_irreps()
        print(f"✓ Irreps calculated: {len(self._irreps)} irreps")

        self._characters, self._irrep_dims = self._get_characters()
        print(f"✓ Characters calculated: {len(self._characters)} character sets")
        print(f"  Character dimensions: {self._irrep_dims}")
        print(
            f"  First character set: {self._characters[0] if self._characters else 'None'}"
        )

        # Get irrep labels
        self._ir_labels = None
        self._RamanIR_labels = None

        if (
            self._pointgroup_symbol in character_table.keys()
            and character_table[self._pointgroup_symbol] is not None
        ):
            print(f"✓ Character table available for {self._pointgroup_symbol}")

            self._rotation_symbols, character_table_of_ptg = self._get_rotation_symbols(
                self._pointgroup_symbol
            )
            self._character_table = character_table_of_ptg
            print(f"✓ Rotation symbols: {self._rotation_symbols[:5]}")
            print(f"  Character table keys: {list(character_table_of_ptg.keys())}")

            if (abs(self._qpoint) < self._symprec).all() and self._rotation_symbols:
                print(f"✓ Q-point is at Gamma, computing irrep labels...")
                self._ir_labels = self._get_irrep_labels(character_table_of_ptg)
                print(f"✓ Irrep labels computed: {self._ir_labels}")
        else:
            print(f"✗ No character table for {self._pointgroup_symbol}")

        return True


# Create instance and run analysis
irreps = _IrRepsDebug(phonopy_atoms, qpoint, freqs, eigvecs, symprec, deg_tol)
success = irreps.run_analysis()

print("\n" + "=" * 80)
print("Analysis Results")
print("=" * 80)

if success:
    print(f"Analysis succeeded: {success}")
    print(f"\nNumber of modes: {len(freqs)}")
    print(f"Number of degenerate sets: {len(irreps._degenerate_sets)}")
    print(f"Number of ir_labels: {len(irreps._ir_labels) if irreps._ir_labels else 0}")

    if irreps._ir_labels:
        print(f"\nIR Labels (indexed by degenerate set):")
        for i, lbl in enumerate(irreps._ir_labels[:20]):  # First 20
            print(f"  Set {i}: {lbl}")

        # Now map to modes
        print(f"\nMapping to modes:")
        mode_to_degset = {}
        for set_idx, deg_set in enumerate(irreps._degenerate_sets):
            for mode_idx in deg_set:
                mode_to_degset[mode_idx] = set_idx

        for mode_idx in range(min(20, len(freqs))):
            set_idx = mode_to_degset.get(mode_idx)
            if set_idx is not None and set_idx < len(irreps._ir_labels):
                label = irreps._ir_labels[set_idx]
                freq = freqs[mode_idx]
                print(
                    f"  Mode {mode_idx}: freq={freq:.2f} THz, set={set_idx}, label={label}"
                )
else:
    print("Analysis failed!")
