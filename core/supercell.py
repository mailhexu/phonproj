"""
Supercell and Phonon Displacement Generation

Provides standalone functions for generating supercell structures from primitive cells
and creating displacement patterns based on phonon modes. These functions enable
the study of phase transitions, lattice dynamics, and structural distortions.

This implementation strictly follows the algorithms from references/cells.py and
references/frozen_mode.py to ensure compatibility and correctness.
"""

import numpy as np
import copy
from typing import Optional, Union, Tuple, List, Callable, Any
from ase import Atoms

# Import from core modules
from phonproj.modes import PhononModes


def default_mod_func(R: np.ndarray) -> np.ndarray:
    """
    Default modulation function returning 1.0 for all positions.
    """
    return np.ones(len(R))


def trim_cell(relative_axes, cell, symprec):
    positions = cell.get_scaled_positions()
    numbers = cell.get_atomic_numbers()
    masses = cell.get_masses()
    lattice = cell.get_cell()
    trimed_lattice = np.dot(relative_axes.T, lattice)

    trimed_positions = []
    trimed_numbers = []
    if masses is None:
        trimed_masses = []
    else:
        trimed_masses = []
    extracted_atoms = []

    positions_in_new_lattice = np.dot(positions, np.linalg.inv(relative_axes).T)
    positions_in_new_lattice -= np.floor(positions_in_new_lattice)
    trimed_positions = np.zeros_like(positions_in_new_lattice)
    num_atom = 0

    mapping_table = np.arange(len(positions), dtype="intc")
    symprec2 = symprec**2
    for i, pos in enumerate(positions_in_new_lattice):
        is_overlap = False
        if num_atom > 0:
            diff = trimed_positions[:num_atom] - pos
            diff -= np.rint(diff)
            distances2 = np.sum(np.dot(diff, trimed_lattice) ** 2, axis=1)
            overlap_indices = np.where(distances2 < symprec2)[0]
            if len(overlap_indices) > 0:
                is_overlap = True
                mapping_table[i] = extracted_atoms[overlap_indices[0]]

        if not is_overlap:
            trimed_positions[num_atom] = pos
            num_atom += 1
            trimed_numbers.append(numbers[i])
            if masses is not None:
                trimed_masses.append(masses[i])
            extracted_atoms.append(i)

    trimed_cell = Atoms(
        numbers=trimed_numbers,
        masses=trimed_masses,
        scaled_positions=trimed_positions[:num_atom],
        cell=trimed_lattice,
    )

    return trimed_cell, extracted_atoms, mapping_table


def determinant(m):
    return (
        m[0][0] * m[1][1] * m[2][2]
        - m[0][0] * m[1][2] * m[2][1]
        + m[0][1] * m[1][2] * m[2][0]
        - m[0][1] * m[1][0] * m[2][2]
        + m[0][2] * m[1][0] * m[2][1]
        - m[0][2] * m[1][1] * m[2][0]
    )


class Supercell(Atoms):
    def __init__(self, unitcell, supercell_matrix, symprec=1e-5):
        self._s2u_map = None
        self._u2s_map = None
        self._u2u_map = None
        self._supercell_matrix = np.array(supercell_matrix, dtype="intc")
        self._create_supercell(unitcell, symprec)

    def get_supercell_matrix(self):
        return self._supercell_matrix

    def get_supercell_to_unitcell_map(self):
        return self._s2u_map

    def get_unitcell_to_supercell_map(self):
        return self._u2s_map

    def get_unitcell_to_unitcell_map(self):
        return self._u2u_map

    def _create_supercell(self, unitcell, symprec):
        mat = self._supercell_matrix
        frame = self._get_surrounding_frame(mat)
        sur_cell, u2sur_map = self._get_simple_supercell(frame, unitcell)

        trim_frame = np.array(
            [
                mat[0] / float(frame[0]),
                mat[1] / float(frame[1]),
                mat[2] / float(frame[2]),
            ]
        )
        supercell, sur2s_map, mapping_table = trim_cell(trim_frame, sur_cell, symprec)

        multi = len(supercell) // len(unitcell)

        if multi != determinant(self._supercell_matrix):
            print("Supercell creation failed.")
            print(
                "Probably some atoms are overwrapped. The mapping table is give below."
            )
            print(mapping_table)
            Atoms.__init__(self)
            self._u2s_map = np.array([], dtype=int)
            self._u2u_map = {}
            self._s2u_map = np.array([], dtype=int)
        else:
            Atoms.__init__(
                self,
                numbers=supercell.get_atomic_numbers(),
                masses=supercell.get_masses(),
                scaled_positions=supercell.get_scaled_positions(),
                cell=supercell.get_cell(),
            )
            self._u2s_map = np.arange(len(unitcell)) * multi
            self._u2u_map = dict([(j, i) for i, j in enumerate(self._u2s_map)])
            self._s2u_map = np.array(u2sur_map)[sur2s_map] * multi
            if self._s2u_map is None:
                self._s2u_map = np.array([], dtype=int)
            if self._u2u_map is None:
                self._u2u_map = {}
            if self._u2s_map is None:
                self._u2s_map = np.array([], dtype=int)

    def _get_surrounding_frame(self, supercell_matrix):
        m = np.array(supercell_matrix)
        axes = np.array(
            [
                [0, 0, 0],
                m[:, 0],
                m[:, 1],
                m[:, 2],
                m[:, 1] + m[:, 2],
                m[:, 2] + m[:, 0],
                m[:, 0] + m[:, 1],
                m[:, 0] + m[:, 1] + m[:, 2],
            ]
        )
        frame = [max(axes[:, i]) - min(axes[:, i]) for i in (0, 1, 2)]
        return frame

    def _get_simple_supercell(self, multi, unitcell):
        positions = unitcell.get_scaled_positions()
        numbers = unitcell.get_atomic_numbers()
        masses = unitcell.get_masses()
        lattice = np.array(unitcell.get_cell())

        atom_map = []
        positions_multi = []
        numbers_multi = []
        masses_multi = []

        for l, pos in enumerate(positions):
            for i in range(multi[2]):
                for j in range(multi[1]):
                    for k in range(multi[0]):
                        positions_multi.append(
                            [
                                (pos[0] + k) / multi[0],
                                (pos[1] + j) / multi[1],
                                (pos[2] + i) / multi[2],
                            ]
                        )
                        numbers_multi.append(numbers[l])
                        if masses is not None:
                            masses_multi.append(masses[l])
                        atom_map.append(l)

        simple_supercell = Atoms(
            numbers=numbers_multi,
            masses=masses_multi,
            scaled_positions=positions_multi,
            cell=np.dot(np.diag(multi), lattice),
        )

        return simple_supercell, atom_map


def generate_supercell(
    primitive_cell: Atoms, supercell_matrix: np.ndarray, symprec: float = 1e-5
) -> Atoms:
    if not isinstance(primitive_cell, Atoms):
        raise TypeError("primitive_cell must be an ASE Atoms object")

    supercell_matrix = np.asarray(supercell_matrix, dtype=int)
    if supercell_matrix.shape != (3, 3):
        raise ValueError("supercell_matrix must be a 3x3 matrix")

    return Supercell(primitive_cell, supercell_matrix, symprec=symprec)


def _get_phase_factor(modulation: np.ndarray, argument: float) -> complex:
    u = np.ravel(modulation)
    index_max_elem = np.argmax(abs(u))
    max_elem = u[index_max_elem]
    phase_for_zero = max_elem / abs(max_elem)
    phase_factor = np.exp(1j * np.pi * argument / 180) / phase_for_zero
    return phase_factor


def _is_valid_mapping(mapping):
    return mapping is not None and hasattr(mapping, "__len__") and len(mapping) > 0


def _get_displacements(
    eigvec: np.ndarray,
    q: np.ndarray,
    amplitude: float,
    argument: float,
    supercell: Atoms,
    mod_func=None,
    use_isotropy_amplitude: bool = True,
    normalize: bool = False,
    n_cells: Optional[int] = None,
) -> np.ndarray:
    if not isinstance(supercell, Supercell):
        raise TypeError(
            "supercell must be a Supercell object for displacement calculation"
        )
    if not (
        _is_valid_mapping(supercell.get_supercell_to_unitcell_map())
        and _is_valid_mapping(supercell.get_unitcell_to_unitcell_map())
    ):
        raise ValueError(
            "Supercell mapping arrays are missing or empty. Supercell creation likely failed."
        )
    if mod_func is None:
        mod_func = default_mod_func

    m = supercell.get_masses()
    s2u_map = supercell.get_supercell_to_unitcell_map()
    u2u_map = supercell.get_unitcell_to_unitcell_map()
    if s2u_map is None or u2u_map is None or len(s2u_map) == 0 or len(u2u_map) == 0:
        raise ValueError(
            "Supercell mapping arrays are missing or empty. Supercell creation likely failed."
        )
    try:
        s2uu_map = [u2u_map[x] for x in s2u_map]
    except Exception as e:
        raise ValueError(f"Error constructing s2uu_map: {e}")

    spos = supercell.get_scaled_positions()
    dim = supercell.get_supercell_matrix()
    r = np.dot(spos, dim.T)

    coefs = np.exp(2j * np.pi * np.dot(r, q)) * mod_func(r)

    u = []
    for i, coef in enumerate(coefs):
        eig_index = s2uu_map[i] * 3
        u.append(eigvec[eig_index : eig_index + 3] * coef)

    u = np.array(u)

    if n_cells is not None and n_cells > 0:
        u /= np.sqrt(n_cells)

    if normalize:
        masses = supercell.get_masses()
        masses_repeated = np.repeat(masses, 3)
        u_flat = u.ravel()
        mass_norm = np.sqrt(np.sum(masses_repeated * np.abs(u_flat) ** 2))
        u /= mass_norm

    phase_factor = _get_phase_factor(u, argument)
    if use_isotropy_amplitude:
        amplitude = amplitude
    u *= phase_factor * amplitude
    return u


def generate_mode_displacement(
    modes: PhononModes,
    q_index: int,
    mode_index: int,
    supercell_matrix: np.ndarray,
    amplitude: float = 0.1,
    argument: float = 0.0,
    mod_func: Optional[Callable] = None,
    use_isotropy_amplitude: bool = True,
    normalize: bool = False,
) -> np.ndarray:
    if not isinstance(modes, PhononModes):
        raise TypeError("modes must be a PhononModes object")

    if q_index < 0 or q_index >= modes.n_qpoints:
        raise ValueError(f"q_index {q_index} out of range [0, {modes.n_qpoints - 1}]")

    if mode_index < 0 or mode_index >= modes.n_modes:
        raise ValueError(
            f"mode_index {mode_index} out of range [0, {modes.n_modes - 1}]"
        )

    supercell_matrix = np.asarray(supercell_matrix, dtype=int)
    if supercell_matrix.shape != (3, 3):
        raise ValueError("supercell_matrix must be a 3x3 matrix")

    frequency, eigenvector = modes.get_mode(q_index, mode_index)
    qpoint = modes.qpoints[q_index]

    supercell = generate_supercell(modes.primitive_cell, supercell_matrix)

    n_cells = len(supercell) // len(modes.primitive_cell)
    displacements_complex = _get_displacements(
        eigvec=eigenvector,
        q=qpoint,
        amplitude=amplitude,
        argument=argument,
        supercell=supercell,
        mod_func=mod_func,
        use_isotropy_amplitude=use_isotropy_amplitude,
        normalize=normalize,
        n_cells=n_cells,
    )

    return displacements_complex.real


def generate_displaced_supercell(
    modes: PhononModes,
    q_index: int,
    mode_index: int,
    supercell_matrix: np.ndarray,
    amplitude: float = 0.1,
    argument: float = 0.0,
    mod_func: Optional[Callable] = None,
    use_isotropy_amplitude: bool = True,
    return_displacements: bool = False,
    normalize: bool = False,
) -> Union[Atoms, Tuple[Atoms, np.ndarray]]:
    supercell = generate_supercell(modes.primitive_cell, supercell_matrix)

    n_cells = len(supercell) // len(modes.primitive_cell)
    displacements_complex = _get_displacements(
        eigvec=modes.get_mode(q_index, mode_index)[1],
        q=modes.qpoints[q_index],
        amplitude=amplitude,
        argument=argument,
        supercell=supercell,
        mod_func=mod_func,
        use_isotropy_amplitude=use_isotropy_amplitude,
        normalize=normalize,
        n_cells=n_cells,
    )

    displacements = displacements_complex.real
    displaced_supercell = copy.deepcopy(supercell)
    positions = displaced_supercell.get_positions()
    positions += displacements
    displaced_supercell.set_positions(positions)

    if return_displacements:
        return displaced_supercell, displacements
    else:
        return displaced_supercell
