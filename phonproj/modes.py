"""
PhononModes - Phonon mode representation and analysis.

Provides classes and functions for representing phonon modes,
generating displaced structures, and analyzing displacement patterns.

Classes:
    PhononModes: Main class for phonon mode representation

Functions:
    create_supercell: Create supercell from unit cell with proper transformations
"""

import copy
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
from ase import Atoms
from ase.build import make_supercell

# Note: PhonBand import moved to where it's needed to avoid circular imports
if TYPE_CHECKING:
    from phonproj.band_structure import PhononBand


class PhononModes:
    """
    Represents phonon modes with structural and vibrational properties.

    This class can handle multiple q-points and multiple phonon modes per q-point,
    providing methods for gauge transformation, displacement generation, and analysis.
    It properly accounts for atomic masses in normalization and handles phase factors
    for gauge transformations.

    Attributes:
        primitive_cell (ase.Atoms): The primitive unit cell structure
        qpoints (np.ndarray): Array of q-points in reciprocal space, shape (n_qpoints, 3)
        frequencies (np.ndarray): Phonon frequencies in THz, shape (n_qpoints, n_modes)
        eigenvectors (np.ndarray): Complex eigenvectors, shape (n_qpoints, n_modes, n_atoms*3)
        atomic_masses (np.ndarray): Atomic masses, shape (n_atoms,)
        gauge (str): Current gauge choice, either "R"  or "r"
    """

    _phonopy_yaml_path: Optional[str] = None
    _phonopy_directory: Optional[str] = None

    def __init__(
        self,
        primitive_cell: Atoms,
        qpoints: np.ndarray,
        frequencies: np.ndarray,
        eigenvectors: np.ndarray,
        atomic_masses: Optional[np.ndarray] = None,
        gauge: str = "R",
    ):
        """
        Initialize a PhononModes object.

        Args:
            primitive_cell: ASE Atoms object representing the primitive unit cell
            qpoints: Array of q-points in reciprocal space, shape (n_qpoints, 3)
            frequencies: Phonon frequencies in THz, shape (n_qpoints, n_modes)
            eigenvectors: Complex eigenvectors, shape (n_qpoints, n_modes, n_atoms*3)
            atomic_masses: Atomic masses, if None uses masses from primitive_cell
            gauge: Gauge choice, either "R"  or "r"

        Raises:
            ValueError: If gauge is not "R" or "r", or if array dimensions are incorrect
            TypeError: If inputs are not of expected types
        """
        # Initialize phonopy YAML path (for phonopy modulation API access)
        self._phonopy_yaml_path: Optional[str] = None
        # Validate inputs
        if not isinstance(primitive_cell, Atoms):
            raise TypeError("primitive_cell must be an ASE Atoms object")

        qpoints = np.asarray(qpoints, dtype=float)
        frequencies = np.asarray(frequencies, dtype=float)
        eigenvectors = np.asarray(eigenvectors, dtype=complex)

        if qpoints.ndim != 2 or qpoints.shape[1] != 3:
            raise ValueError("qpoints must be shape (n_qpoints, 3)")

        if frequencies.ndim != 2:
            raise ValueError("frequencies must be 2D array")

        if eigenvectors.ndim != 3:
            raise ValueError("eigenvectors must be 3D array")

        if (
            qpoints.shape[0] != frequencies.shape[0]
            or qpoints.shape[0] != eigenvectors.shape[0]
        ):
            raise ValueError(
                "First dimension of qpoints, frequencies, and eigenvectors must match"
            )

        if frequencies.shape[1] != eigenvectors.shape[1]:
            raise ValueError(
                "Second dimension of frequencies and eigenvectors must match"
            )

        if gauge not in ["R", "r"]:
            raise ValueError("gauge must be either 'R' or 'r'")

        n_atoms = len(primitive_cell)
        if eigenvectors.shape[2] != n_atoms * 3:
            raise ValueError(
                f"Third dimension of eigenvectors must be {n_atoms * 3} "
                f"(3 components × {n_atoms} atoms)"
            )

        # Store attributes
        self.primitive_cell = primitive_cell
        self.qpoints = qpoints
        self.frequencies = frequencies
        self.eigenvectors = eigenvectors
        self.gauge = gauge
        self._n_atoms = n_atoms
        self._n_qpoints = int(qpoints.shape[0])
        self._n_modes = int(frequencies.shape[1])

        # Get atomic masses
        if atomic_masses is None:
            self.atomic_masses = primitive_cell.get_masses()
        else:
            self.atomic_masses = np.asarray(atomic_masses)

        if len(self.atomic_masses) != n_atoms:
            raise ValueError("Number of atomic masses must match number of atoms")

    @property
    def n_qpoints(self) -> int:
        """Number of q-points."""
        return int(self._n_qpoints)

    @property
    def n_modes(self) -> int:
        """Number of phonon modes per q-point."""
        return self._n_modes

    @property
    def n_atoms(self) -> int:
        """Number of atoms in unit cell."""
        return self._n_atoms

    def __repr__(self) -> str:
        """Return string representation of the PhononModes object."""
        return (
            f"PhononModes(n_qpoints={self._n_qpoints}, n_modes={self._n_modes}, "
            f"n_atoms={self._n_atoms}, gauge='{self.gauge}')"
        )

    def transform_gauge(self, new_gauge: str) -> "PhononModes":
        """
        Transform between "R" and "r" gauges.

        The R and r gauges differ by a phase factor that depends on the q-point
        and atomic positions. This function applies the appropriate phase transformation.

        Args:
            new_gauge: Target gauge, either "R" or "r"

        Returns:
            PhononModes: New PhononModes object with transformed gauge

        Raises:
            ValueError: If new_gauge is not "R" or "r"
        """
        if new_gauge not in ["R", "r"]:
            raise ValueError("new_gauge must be either 'R' or 'r'")

        if new_gauge == self.gauge:
            # No transformation needed
            return copy.deepcopy(self)

        # Get scaled positions for phase factor calculation
        scaled_positions = self.primitive_cell.get_scaled_positions()

        # Apply phase transformation to all eigenvectors
        transformed_eigenvectors = np.zeros_like(self.eigenvectors)

        for q_idx, qpoint in enumerate(self.qpoints):
            for mode_idx in range(self._n_modes):
                eigvec = self.eigenvectors[q_idx, mode_idx]

                # Apply phase factors: exp(±2πi * q · r)
                # + for R → r, - for r → R
                sign = 1 if self.gauge == "R" else -1

                # Create phase factors for each atom
                phases = np.exp(sign * 2j * np.pi * np.dot(scaled_positions, qpoint))
                # Repeat for each Cartesian direction
                phases = np.repeat(phases, 3)

                # Apply phase transformation
                transformed_eigenvectors[q_idx, mode_idx] = eigvec * phases

        return PhononModes(
            primitive_cell=self.primitive_cell,
            qpoints=self.qpoints,
            frequencies=self.frequencies,
            eigenvectors=transformed_eigenvectors,
            atomic_masses=self.atomic_masses,
            gauge=new_gauge,
        )

    def get_mode(self, q_index: int, mode_index: int) -> Tuple[float, np.ndarray]:
        """
        Extract a specific phonon mode.

        Args:
            q_index: Index of the q-point
            mode_index: Index of the phonon mode

        Returns:
            Tuple[float, np.ndarray]: (frequency, eigenvector) for the specified mode

        Raises:
            ValueError: If indices are out of range
        """
        if q_index < 0 or q_index >= self._n_qpoints:
            raise ValueError(
                f"q_index {q_index} out of range [0, {self._n_qpoints - 1}]"
            )

        if mode_index < 0 or mode_index >= self._n_modes:
            raise ValueError(
                f"mode_index {mode_index} out of range [0, {self._n_modes - 1}]"
            )

        frequency = self.frequencies[q_index, mode_index]
        eigenvector = self.eigenvectors[q_index, mode_index]

        return frequency, eigenvector

    def get_eigen_displacement(
        self, q_index: int, mode_index: int, normalize: bool = True
    ) -> np.ndarray:
        """
        Extract the eigen displacement pattern for a specific phonon mode.

        Eigen displacements are always mass-weighted for physical correctness,
        following the convention used in phonon calculations.

        Since eigenvectors from phonopy are already orthonormal, we keep them as
        complex values for non-Gamma points. For Gamma points, eigenvectors are
        real by symmetry.

        Args:
            q_index: Index of the q-point
            mode_index: Index of the phonon mode
            normalize: If True, return normalized displacement vectors

        Returns:
            numpy.ndarray: Mass-weighted displacement vectors of shape (n_atoms, 3)
                          Complex for non-Gamma q-points, real for Gamma point

        Raises:
            ValueError: If indices are out of range
        """
        frequency, eigenvector = self.get_mode(q_index, mode_index)

        # Reshape from flat vector to (n_atoms, 3) array
        displacement = eigenvector.reshape(self._n_atoms, 3)

        # Check if this is Gamma point (all components close to zero)
        qpoint = self.qpoints[q_index]
        is_gamma = np.allclose(qpoint, 0.0, atol=1e-6)

        if is_gamma:
            # For Gamma point, eigenvectors should be real
            displacement = displacement.real

        # For non-Gamma points, keep complex values - eigenvectors are already orthonormal
        # No need for additional orthogonalization

        # Apply mass-weighting (always required for eigen displacements)
        # Standard phonon theory: u_i = e_i / sqrt(m) (NOT e_i * sqrt(m))
        mass_weights = np.sqrt(self.atomic_masses)
        displacement = displacement / mass_weights[:, np.newaxis]

        if normalize:
            # Use mass-weighted normalization for eigen displacements
            norm = self.mass_weighted_norm(displacement)
            if norm > 1e-12:  # Avoid division by zero
                displacement /= norm

        return displacement

    def generate_full_commensurate_grid(
        self, supercell_matrix: np.ndarray
    ) -> List[np.ndarray]:
        """
        Generate the full set of commensurate q-vectors implied by an integer diagonal
        supercell matrix. For a diagonal supercell S = diag(sx, sy, sz) this returns
        q = (i/sx, j/sy, k/sz) for i=0..sx-1, etc.

        This helper currently supports diagonal integer supercells. For non-diagonal
        supercell matrices the function raises NotImplementedError.
        """
        S = np.asarray(supercell_matrix)
        if S.shape != (3, 3):
            raise ValueError("supercell_matrix must be a 3x3 matrix")

        # Check for diagonal integer supercell
        if not np.allclose(S, np.diag(np.diag(S))):
            raise NotImplementedError(
                "Full commensurate-grid generation currently supports only diagonal integer supercells"
            )

        sx, sy, sz = np.diag(S).astype(int)
        if sx <= 0 or sy <= 0 or sz <= 0:
            raise ValueError("Supercell diagonal entries must be positive integers")

        q_vectors = []
        for i in range(sx):
            for j in range(sy):
                for k in range(sz):
                    q = np.array([i / sx, j / sy, k / sz], dtype=float)
                    # Normalize into [0, 1) canonical representation
                    q = q - np.floor(q)
                    q_vectors.append(q)

        return q_vectors

    def get_commensurate_qpoints(
        self,
        supercell_matrix: np.ndarray,
        tolerance: float = 1e-6,
        detailed: bool = False,
    ) -> Union[List[int], Dict[str, Any]]:
        """
        Find which of the existing `self.qpoints` correspond to the full set of
        commensurate q-points implied by `supercell_matrix`.

        Backwards-compatible behavior (default): return a list of matched indices
        (same as legacy callers expected).

        If `detailed=True`, return a dict with keys:
         - 'matched_indices': list of integer indices into `self.qpoints` that were found
         - 'missing_qpoints': list of q-vectors (np.ndarray) that are expected but not present
         - 'all_qpoints': list of all expected q-vectors (np.ndarray)

        This prevents callers from silently working with only a subset of q-points
        while preserving compatibility with existing code.
        """
        supercell_matrix = np.asarray(supercell_matrix)
        if supercell_matrix.shape != (3, 3):
            raise ValueError("supercell_matrix must be a 3x3 matrix")

        # Generate canonical list of expected commensurate q-vectors
        try:
            all_qpoints = self.generate_full_commensurate_grid(supercell_matrix)
        except NotImplementedError:
            # Fall back to scanning existing qpoints for commensurability
            matched = []
            for q_index, qpoint in enumerate(self.qpoints):
                try:
                    self._check_qpoint_supercell_commensurability(
                        qpoint, supercell_matrix, tolerance
                    )
                    matched.append(q_index)
                except ValueError:
                    continue

            if detailed:
                return {
                    "matched_indices": matched,
                    "missing_qpoints": [],
                    "all_qpoints": [],
                }
            else:
                return matched

        # Match expected q-vectors to those available in self.qpoints
        matched_indices = []
        missing_qpoints = []

        # Build a lookup by converting to rounded tuples according to tolerance
        decimals = int(-np.log10(tolerance)) if tolerance < 1 else 8
        q_lookup = {}
        for idx, q in enumerate(self.qpoints):
            key = tuple(np.round(q % 1.0, decimals))
            q_lookup[key] = idx

        for q in all_qpoints:
            # Normalize to canonical representative in [0,1)
            q_norm = q % 1.0
            key = tuple(np.round(q_norm, decimals))
            if key in q_lookup:
                matched_indices.append(q_lookup[key])
            else:
                # Try a tolerant search in case of floating rounding differences
                found = False
                for idx, q_existing in enumerate(self.qpoints):
                    if np.allclose(q_existing % 1.0, q_norm, atol=tolerance):
                        matched_indices.append(idx)
                        found = True
                        break
                if not found:
                    missing_qpoints.append(q_norm)

        if detailed:
            return {
                "matched_indices": matched_indices,
                "missing_qpoints": missing_qpoints,
                "all_qpoints": all_qpoints,
            }
        else:
            return matched_indices

    def generate_all_mode_displacements(
        self, q_index: int, supercell_matrix: np.ndarray, amplitude: float = 1.0
    ) -> np.ndarray:
        """
        Generate displacement patterns for all phonon modes at a specific q-point in a supercell.

        This method generates the complete set of displacement patterns for all phonon modes
        at a given q-point, creating an orthonormal basis for that specific q-point's contribution
        to the supercell displacement space.

        Args:
            q_index: Index of the q-point
            supercell_matrix: 3x3 supercell transformation matrix
            amplitude: Displacement amplitude for normalization (default: 1.0)

        Returns:
            np.ndarray: Array of displacement patterns with shape (n_modes, n_supercell_atoms, 3)
                       Each displacement pattern has unit mass-weighted norm.
                       Array dtype is complex to preserve eigenvector phase information.

        Raises:
            ValueError: If q_index is out of range or q-point is not commensurate

        Examples:
            >>> # Generate all mode displacements for Gamma point in 1x1x1 supercell
            >>> gamma_displacements = modes.generate_all_mode_displacements(
            ...     q_index=0, supercell_matrix=np.eye(3), amplitude=1.0
            ... )
            >>> print(f"Generated {gamma_displacements.shape[0]} mode displacements")
        """
        if q_index < 0 or q_index >= self.n_qpoints:
            raise ValueError(
                f"q_index {q_index} out of range [0, {self.n_qpoints - 1}]"
            )

        # Check if q-point is commensurate with supercell
        qpoint = self.qpoints[q_index]
        self._check_qpoint_supercell_commensurability(qpoint, supercell_matrix)

        # Calculate number of supercell atoms
        det = int(round(abs(np.linalg.det(supercell_matrix))))
        n_supercell_atoms = det * self.n_atoms

        # Check if this is the Gamma point and 1x1x1 supercell case
        is_gamma = np.allclose(qpoint, 0.0, atol=1e-8)
        is_unit_supercell = np.allclose(supercell_matrix, np.eye(3), atol=1e-8)

        if is_gamma and is_unit_supercell:
            # Special case: Gamma point with 1x1x1 supercell
            # Eigenvectors are orthonormal under plain inner product,
            # need to transform them to be orthonormal under mass-weighted inner product

            eigenvectors = self.eigenvectors[q_index]  # Shape: (n_modes, n_atoms*3)
            masses = self.atomic_masses
            masses_repeated = np.repeat(masses, 3)  # Shape: (n_atoms*3,)

            # Transform eigenvectors to be orthonormal under mass-weighted inner product
            # Scale each component by 1/sqrt(mass) and then normalize
            all_displacements = np.zeros(
                (self.n_modes, n_supercell_atoms, 3), dtype=complex
            )

            for mode_index in range(self.n_modes):
                # Get the eigenvector (flattened)
                eigvec_flat = eigenvectors[mode_index]

                # Scale by 1/sqrt(mass) to convert from plain to mass-weighted orthonormality
                scaled_eigvec = eigvec_flat / np.sqrt(masses_repeated)

                # Normalize under mass-weighted norm
                mass_weighted_norm_sq = np.sum(
                    masses_repeated * np.abs(scaled_eigvec) ** 2
                )
                mass_weighted_norm = np.sqrt(mass_weighted_norm_sq)
                normalized_eigvec = scaled_eigvec / mass_weighted_norm

                # Apply amplitude scaling
                final_eigvec = normalized_eigvec * amplitude

                # Apply phase normalization: make the maximum component real and positive
                index_max_elem = np.argmax(np.abs(final_eigvec))
                max_elem = final_eigvec[index_max_elem]
                phase_for_zero = max_elem / np.abs(max_elem)
                phase_factor = 1.0 / phase_for_zero
                final_eigvec = final_eigvec * phase_factor

                # Reshape to (n_atoms, 3) - keep complex values
                all_displacements[mode_index] = final_eigvec.reshape(
                    n_supercell_atoms, 3
                )
        else:
            # General case: Apply the same eigenvector transformation for all cases
            # Eigenvectors are orthonormal under plain inner product,
            # need to transform them to be orthonormal under mass-weighted inner product

            all_displacements = np.zeros(
                (self.n_modes, n_supercell_atoms, 3), dtype=complex
            )

            for mode_index in range(self.n_modes):
                # First get the displacement using the local method but WITHOUT normalization
                # to preserve the original eigenvector relationships
                displacement = self._calculate_supercell_displacements(
                    q_index=q_index,
                    mode_index=mode_index,
                    supercell_matrix=supercell_matrix,
                    n_supercell_atoms=n_supercell_atoms,
                    phase=0.0,
                    amplitude=1.0,  # Don't apply amplitude yet
                    take_real=True,  # Keep complex for orthonormal basis
                )

                # Get supercell masses for this displacement
                supercell_masses = np.tile(self.atomic_masses, det)

                # The displacement from _calculate_supercell_displacements is already a properly
                # mass-weighted physical displacement vector. We just need to normalize
                # it to unit mass-weighted norm for the orthonormal basis.
                displacement_flat = displacement.flatten()
                masses_repeated = np.repeat(supercell_masses, 3)

                # Normalize under mass-weighted norm with higher precision
                mass_weighted_norm_sq = np.sum(
                    masses_repeated * np.abs(displacement_flat) ** 2, dtype=np.float64
                )
                mass_weighted_norm = np.sqrt(mass_weighted_norm_sq)

                # Avoid division by tiny numbers
                if mass_weighted_norm < 1e-14:
                    raise ValueError(f"Zero norm encountered for mode {mode_index}")

                normalized_displacement = displacement_flat / mass_weighted_norm

                # Apply amplitude scaling
                final_displacement = normalized_displacement * amplitude

                # Apply phase normalization: make the maximum component real and positive
                index_max_elem = np.argmax(np.abs(final_displacement))
                max_elem = final_displacement[index_max_elem]
                phase_for_zero = max_elem / np.abs(max_elem)
                phase_factor = 1.0 / phase_for_zero
                final_displacement = final_displacement * phase_factor

                # Reshape back - keep complex values
                all_displacements[mode_index] = final_displacement.reshape(
                    n_supercell_atoms, 3
                )

        return all_displacements

    def generate_all_commensurate_displacements(
        self, supercell_matrix: np.ndarray, amplitude: float = 1.0
    ) -> Dict[int, np.ndarray]:
        """
        Generate displacement patterns for all commensurate q-points of a given supercell.

        This method creates the complete orthonormal basis for the supercell displacement space
        by generating displacement patterns for all q-points that are commensurate with the
        supercell matrix. According to Bloch's theorem, these form a complete orthonormal basis.

        Args:
            supercell_matrix: 3x3 supercell transformation matrix
            amplitude: Displacement amplitude for normalization (default: 1.0)

        Returns:
            Dict[int, np.ndarray]: Dictionary mapping q-point indices to their displacement arrays.
                                 Each displacement array has shape (n_modes, n_supercell_atoms, 3)
                                 and each displacement pattern has unit mass-weighted norm.

        Raises:
            ValueError: If supercell_matrix is not a valid 3x3 matrix

        Examples:
            >>> # Generate all commensurate displacements for 2x2x2 supercell
            >>> supercell_matrix = np.eye(3) * 2
            >>> all_displacements = modes.generate_all_commensurate_displacements(supercell_matrix)
            >>> print(f"Generated displacements for {len(all_displacements)} q-points")
            >>>
            >>> # Each q-point contributes an orthonormal set of mode displacements
            >>> for q_index, displacements in all_displacements.items():
            ...     print(f"Q-point {q_index}: {displacements.shape[0]} modes")
        """
        supercell_matrix = np.asarray(supercell_matrix)
        if supercell_matrix.shape != (3, 3):
            raise ValueError("supercell_matrix must be a 3x3 matrix")

        # Get all commensurate q-points (detailed info)
        result = self.get_commensurate_qpoints(supercell_matrix, detailed=True)

        # Type assertion: detailed=True should return dict, but handle gracefully
        assert isinstance(
            result, dict
        ), f"Expected dict from get_commensurate_qpoints(detailed=True), got {type(result)}"

        matched_indices = result.get("matched_indices", [])
        missing_qpoints = result.get("missing_qpoints", [])
        all_qpoints = result.get("all_qpoints", [])

        if len(matched_indices) == 0:
            raise ValueError(
                f"No commensurate q-points found for supercell matrix:\n{supercell_matrix}\n"
                f"Available q-points: {self.qpoints.tolist()}"
            )

        if len(missing_qpoints) > 0:
            # Provide a helpful error listing missing q-vectors and suggested actions
            raise ValueError(
                f"Missing commensurate q-points for supercell {np.diag(supercell_matrix)}:"
                f" {len(missing_qpoints)} q-points are expected but not present in the PhononModes object.\n"
                f"Missing q-vectors (reciprocal coordinates): {missing_qpoints}\n"
                f"All expected q-vectors: {all_qpoints}\n"
                f"Present q-vectors: {self.qpoints.tolist()}\n"
                f"Suggested actions: generate phonon modes for the missing q-points, or use a supercell"
                f" that matches the available q-mesh."
            )

        # Generate displacements for each commensurate q-point
        all_commensurate_displacements = {}

        for q_index in matched_indices:
            displacements = self.generate_all_mode_displacements(
                q_index=q_index, supercell_matrix=supercell_matrix, amplitude=amplitude
            )
            all_commensurate_displacements[q_index] = displacements

        return all_commensurate_displacements

    def _check_qpoint_supercell_commensurability(
        self, qpoint: np.ndarray, supercell_matrix: np.ndarray, tolerance: float = 1e-6
    ) -> None:
        """
        Check if a q-point is commensurate with the given supercell.

        A q-point q is commensurate with a supercell matrix S if S^T @ q has integer components
        (within numerical tolerance). This ensures that the phase factor exp(2πi q·R) is periodic
        over the supercell lattice vectors.

        Args:
            qpoint: q-point in reciprocal lattice coordinates
            supercell_matrix: 3x3 supercell transformation matrix
            tolerance: Numerical tolerance for checking integer values

        Raises:
            ValueError: If the q-point is not commensurate with the supercell
        """
        # Calculate S^T @ q
        transformed_q = supercell_matrix.T @ qpoint

        # Check if all components are close to integers
        fractional_parts = transformed_q - np.round(transformed_q)

        if np.any(np.abs(fractional_parts) > tolerance):
            # Format the supercell dimensions for clear error message
            if np.allclose(supercell_matrix, np.diag(np.diag(supercell_matrix))):
                # Diagonal matrix - extract dimensions
                dims = np.diag(supercell_matrix).astype(int)
                supercell_desc = f"{dims[0]}×{dims[1]}×{dims[2]}"
            else:
                # Non-diagonal matrix
                supercell_desc = f"matrix:\n{supercell_matrix}"

            raise ValueError(
                f"Q-point {qpoint} is not commensurate with the {supercell_desc} supercell.\n"
                f"For a q-point to be commensurate, S^T @ q must have integer components.\n"
                f"Got S^T @ q = {transformed_q}, fractional parts = {fractional_parts}\n"
                f"Consider using a larger supercell or a different q-point."
            )

    def _calculate_supercell_displacements(
        self,
        q_index: int,
        mode_index: int,
        supercell_matrix: np.ndarray,
        n_supercell_atoms: int,
        phase: float = 0.0,
        amplitude: float = 1.0,
        take_real: bool = True,
    ) -> np.ndarray:
        """
        Calculate displacement vectors for all atoms in the supercell.

        Uses mass-weighted eigen displacements for physically correct displacement patterns.

        Args:
            q_index: Index of the q-point
            mode_index: Index of the phonon mode
            supercell_matrix: 3x3 supercell transformation matrix
            n_supercell_atoms: Number of atoms in the supercell
            phase: Phase angle in radians to apply to the displacement (default: 0.0)
            amplitude: Target amplitude for the final displacement mass-weighted norm (default: 1.0)
            take_real: Whether to take the real part of displacements (default: True)

        Returns:
            numpy.ndarray: Mass-weighted displacement vectors of shape (n_supercell_atoms, 3)
        """
        # Get the mode eigenvector
        _, eigenvector = self.get_mode(q_index, mode_index)
        qpoint = self.qpoints[q_index]

        # Check if q-point is commensurate with supercell
        self._check_qpoint_supercell_commensurability(qpoint, supercell_matrix)

        # Get scaled positions of primitive cell
        prim_scaled_pos = self.primitive_cell.get_scaled_positions()

        # Calculate supercell to unit cell mapping
        # For simplicity, we'll assume a straightforward mapping
        det = int(round(np.linalg.det(supercell_matrix)))

        # Initialize displacement array
        displacements = np.zeros(
            (n_supercell_atoms, 3), dtype=complex if not take_real else float
        )

        # Calculate phase factors and displacements for each supercell atom
        for i in range(n_supercell_atoms):
            # Map supercell atom to primitive cell atom
            prim_atom_index = i % self._n_atoms
            supercell_replica = i // self._n_atoms

            # Calculate lattice vector R for this supercell atom
            # For a 2x2x2 supercell, we need proper 3D lattice vector mapping
            # Extract supercell dimensions from the supercell matrix
            if np.allclose(supercell_matrix, np.diag(np.diag(supercell_matrix))):
                # Diagonal supercell matrix (simple case)
                nx, ny = (
                    int(supercell_matrix[0, 0]),
                    int(supercell_matrix[1, 1]),
                )
                _ = int(supercell_matrix[2, 2])  # nz not used
            else:
                # For non-diagonal matrices, we'd need more complex logic
                # For now, assume 2x2x2 based on determinant
                det_cbrt = round(det ** (1 / 3))
                nx = ny = det_cbrt

            # Map supercell atom to proper lattice vector
            # Each unit cell has self._n_atoms atoms, so:
            unit_cell_index = supercell_replica

            # Convert unit cell index to 3D lattice coordinates
            ix = unit_cell_index % nx
            iy = (unit_cell_index // nx) % ny
            iz = unit_cell_index // (nx * ny)

            lattice_vector = np.array([ix, iy, iz], dtype=float)

            # Calculate phase factor: exp(2πi * q · (r + R)) * exp(i*phase)
            qpoint_phase = np.exp(
                2j
                * np.pi
                * np.dot(qpoint, prim_scaled_pos[prim_atom_index] + lattice_vector)
            ) * np.exp(1j * phase)

            # Get the eigenvector components for this atom
            start_idx = prim_atom_index * 3
            end_idx = start_idx + 3
            atom_eigenvector = eigenvector[start_idx:end_idx]

            # Apply phase factor
            displacement = atom_eigenvector * qpoint_phase

            # For supercell displacements, take the real part
            # The phase factor exp(2πi q·R) creates the proper spatial pattern,
            # and the real part gives the actual physical displacement
            if take_real:
                displacement = displacement.real

            # Apply mass-normalization: u_i = e_i / sqrt(m) (NOT e_i * sqrt(m))
            displacement /= np.sqrt(self.atomic_masses[prim_atom_index])

            displacements[i] = displacement

        # Apply amplitude scaling to achieve target mass-weighted norm
        # For supercells, target norm should be amplitude/N where N is the number of primitive cells
        det = int(round(np.linalg.det(supercell_matrix)))
        target_norm = amplitude / det

        current_norm = self.mass_weighted_norm(displacements)
        if current_norm > 1e-12:  # Avoid division by zero
            displacements = displacements * target_norm / current_norm

        return displacements

    def _calculate_supercell_displacements_phonopy(
        self,
        q_index: int,
        mode_index: int,
        supercell_matrix: np.ndarray,
        amplitude: float = 0.1,
        argument: float = 0.0,
        mod_func: Optional[Callable] = None,
        use_isotropy_amplitude: bool = True,
        normalize: bool = False,
    ) -> np.ndarray:
        """
        Compatibility implementation expected by core.supercell.generate_mode_displacement.

        Uses the local `_get_displacements` + `generate_supercell` helpers so callers
        that expect a phonopy-backed method can call this without causing an
        AttributeError. This implementation mirrors the fallback in
        `phonproj.core.supercell.generate_mode_displacement`.
        """
        from phonproj.core.supercell import _get_displacements, generate_supercell

        # Validate indices
        if q_index < 0 or q_index >= self.n_qpoints:
            raise ValueError(
                f"q_index {q_index} out of range [0, {self.n_qpoints - 1}]"
            )

        if mode_index < 0 or mode_index >= self.n_modes:
            raise ValueError(
                f"mode_index {mode_index} out of range [0, {self.n_modes - 1}]"
            )

        supercell_matrix = np.asarray(supercell_matrix, dtype=int)
        if supercell_matrix.shape != (3, 3):
            raise ValueError("supercell_matrix must be a 3x3 matrix")

        # Extract mode data
        frequency, eigenvector = self.get_mode(q_index, mode_index)
        qpoint = self.qpoints[q_index]

        # Ensure q-point is commensurate
        self._check_qpoint_supercell_commensurability(qpoint, supercell_matrix)

        # Build supercell and compute displacements using the local helper
        supercell = generate_supercell(self.primitive_cell, supercell_matrix)
        n_cells = len(supercell) // len(self.primitive_cell)

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

        return displacements_complex

    def mass_weighted_norm(
        self, displacement: np.ndarray, atomic_masses: Optional[np.ndarray] = None
    ) -> float:
        """
        Calculate mass-weighted norm of a displacement pattern.

        For mass-normalized eigen displacements, the norm should be:
        ||d||_M = sqrt(Σ m * |d|²)

        Args:
            displacement: Displacement array of shape (n_atoms, 3) or flattened
            atomic_masses: Optional custom atomic masses array. For supercells,
                           provide the actual supercell masses array of length n_atoms.

        Returns:
            float: Mass-weighted norm
        """
        if displacement.ndim == 1:
            # Already flattened
            displacement_flat = displacement
            n_atoms = len(displacement_flat) // 3
        else:
            # Flatten from (n_atoms, 3) to (n_atoms*3,)
            displacement_flat = displacement.ravel()
            n_atoms = len(displacement)

        # Use provided masses or default to primitive cell masses
        if atomic_masses is None:
            atomic_masses = self.atomic_masses

        # Check if masses match the displacement size
        if len(atomic_masses) == n_atoms:
            # Masses match the number of atoms - use directly
            masses_repeated = np.repeat(atomic_masses, 3)
        elif len(atomic_masses) == self.n_atoms:
            # Primitive cell masses provided for supercell displacement
            # This is problematic but we'll handle it for backward compatibility
            n_repeats = n_atoms // len(atomic_masses)
            if n_atoms % len(atomic_masses) != 0:
                raise ValueError(
                    f"Cannot map {len(atomic_masses)} primitive masses to {n_atoms} atoms. "
                    f"Provide explicit supercell masses or use a different approach."
                )
            masses_repeated = np.repeat(np.tile(atomic_masses, n_repeats), 3)
        else:
            raise ValueError(
                f"Masses length ({len(atomic_masses)}) doesn't match number of atoms ({n_atoms}) "
                f"or primitive cell atoms ({self.n_atoms})."
            )

        # Calculate mass-weighted norm (use abs for complex displacements)
        return float(np.sqrt(np.sum(masses_repeated * np.abs(displacement_flat) ** 2)))

    def mass_weighted_projection(
        self,
        displacement1: np.ndarray,
        displacement2: np.ndarray,
        atomic_masses: Optional[np.ndarray] = None,
        use_real_part: bool = False,
        debug: bool = False,
    ) -> complex:
        """
        Calculate mass-weighted inner product (projection) between two displacements.

        For mass-normalized eigen displacements, the projection is:
        <d1|d2>_M = Σ m * d1* · d2

        Args:
            displacement1: First displacement array
            displacement2: Second displacement array
            atomic_masses: Optional custom atomic masses array. For supercells,
                           provide the actual supercell masses array of length n_atoms.
            use_real_part: If True, use only real parts of displacements for projection
            debug: If True, print debug information about the calculation

        Returns:
            complex: Mass-weighted projection coefficient
        """
        if displacement1.ndim == 1:
            disp1_flat = displacement1
        else:
            disp1_flat = displacement1.ravel()

        if displacement2.ndim == 1:
            disp2_flat = displacement2
        else:
            disp2_flat = displacement2.ravel()

        # Ensure both displacements have the same length
        if len(disp1_flat) != len(disp2_flat):
            raise ValueError(
                f"Displacement lengths don't match: {len(disp1_flat)} vs {len(disp2_flat)}"
            )

        # Determine number of atoms
        n_atoms = len(disp1_flat) // 3

        # Use provided masses or default to primitive cell masses
        if atomic_masses is None:
            atomic_masses = self.atomic_masses

        # Check if masses match the displacement size
        if len(atomic_masses) == n_atoms:
            # Masses match the number of atoms - use directly
            masses_repeated = np.repeat(atomic_masses, 3)
        elif len(atomic_masses) == self.n_atoms:
            # Primitive cell masses provided for supercell displacement
            # This is problematic but we'll handle it for backward compatibility
            n_repeats = n_atoms // len(atomic_masses)
            if n_atoms % len(atomic_masses) != 0:
                raise ValueError(
                    f"Cannot map {len(atomic_masses)} primitive masses to {n_atoms} atoms. "
                    f"Provide explicit supercell masses or use a different approach."
                )
            masses_repeated = np.repeat(np.tile(atomic_masses, n_repeats), 3)
        else:
            raise ValueError(
                f"Masses length ({len(atomic_masses)}) doesn't match number of atoms ({n_atoms}) "
                f"or primitive cell atoms ({self.n_atoms})."
            )

        # Apply real part filter if requested
        if use_real_part:
            disp1_flat = disp1_flat.real
            disp2_flat = disp2_flat.real

        # Calculate mass-weighted inner product
        projection = complex(np.sum(masses_repeated * np.conj(disp1_flat) * disp2_flat))

        if debug:
            print("DEBUG: mass_weighted_projection")
            print(f"  disp1 shape: {disp1_flat.shape}, disp2 shape: {disp2_flat.shape}")
            print(f"  use_real_part: {use_real_part}")
            print(f"  masses_repeated shape: {masses_repeated.shape}")
            print(f"  projection: {projection}")
            print(f"  |projection|: {abs(projection)}")

        return projection

    def mass_weighted_projection_coefficient(
        self,
        eigenvector: np.ndarray,
        target_displacement: np.ndarray,
        atomic_masses: Optional[np.ndarray] = None,
        use_real_part: bool = False,
        debug: bool = False,
    ) -> complex:
        """
        Calculate mass-weighted projection coefficient for an eigenvector onto a target displacement.

        This properly handles mass-normalized eigen displacements:
        c = <e|d>_M / (||e||_M * ||d||_M)

        Args:
            eigenvector: Mass-normalized eigen displacement eigenvector
            target_displacement: Target displacement pattern
            atomic_masses: Optional custom atomic masses array (for supercells)
            use_real_part: If True, use only real parts of displacements for projection
            debug: If True, print debug information about the calculation

        Returns:
            complex: Projection coefficient
        """
        # Calculate mass-weighted norms
        eigenvector_norm = self.mass_weighted_norm(eigenvector, atomic_masses)
        target_norm = self.mass_weighted_norm(target_displacement, atomic_masses)

        # Calculate mass-weighted projection
        projection = self.mass_weighted_projection(
            eigenvector, target_displacement, atomic_masses, use_real_part, debug
        )

        # Return normalized coefficient
        if eigenvector_norm > 0 and target_norm > 0:
            coefficient = projection / (eigenvector_norm * target_norm)

            if debug:
                print("DEBUG: mass_weighted_projection_coefficient")
                print(f"  eigenvector_norm: {eigenvector_norm}")
                print(f"  target_norm: {target_norm}")
                print(f"  projection: {projection}")
                print(f"  coefficient: {coefficient}")
                print(f"  |coefficient|: {abs(coefficient)}")

            return coefficient
        else:
            if debug:
                print("DEBUG: mass_weighted_projection_coefficient")
                print(
                    f"  Warning: eigenvector_norm={eigenvector_norm}, target_norm={target_norm}"
                )
                print("  Returning 0.0 due to zero norm")
            return 0.0

    def check_eigenvector_orthonormality(
        self, q_index: int, tolerance: float = 1e-10, verbose: bool = False
    ) -> Tuple[bool, float, Dict[str, float]]:
        """
        Check if eigenvectors at a specific q-point are orthonormal.

        This method verifies that eigenvectors form an orthonormal basis, which is
        a fundamental property of the dynamical matrix eigenvalue problem.

        The orthonormality condition is:
            <e_i|e_j> = e_i† @ e_j = δ_ij

        where δ_ij is the Kronecker delta (1 if i=j, 0 otherwise).

        Parameters
        ----------
        q_index : int
            Index of the q-point to check
        tolerance : float, optional
            Numerical tolerance for orthonormality check. Default is 1e-10
        verbose : bool, optional
            If True, print detailed information about the check

        Returns
        -------
        Tuple[bool, float, Dict[str, float]]
            - bool: True if eigenvectors are orthonormal, False otherwise
            - float: Maximum deviation from identity matrix
            - Dict: Dictionary with detailed error metrics

        Raises
        ------
        ValueError
            If q_index is out of range

        Examples
        --------
        >>> modes = PhononModes(...)
        >>> is_orthonormal, max_error, errors = modes.check_eigenvector_orthonormality(0)
        >>> print(f"Orthonormal: {is_orthonormal}, Max error: {max_error:.2e}")
        """
        if q_index < 0 or q_index >= self.n_qpoints:
            raise ValueError(
                f"q_index {q_index} out of range [0, {self.n_qpoints - 1}]"
            )

        if verbose:
            print(f"\nChecking eigenvector orthonormality at q-point index: {q_index}")

        n_modes = self.n_modes

        # Get eigenvectors at the specified q-point
        eigenvectors_q = self.eigenvectors[q_index]  # Shape: (n_modes, n_atoms*3)

        # Calculate inner products: <e_i|e_j> = e_i† @ e_j
        inner_products = np.zeros((n_modes, n_modes), dtype=complex)

        for i in range(n_modes):
            for j in range(n_modes):
                inner_products[i, j] = np.vdot(eigenvectors_q[i], eigenvectors_q[j])

        # Check if inner products form identity matrix
        identity = np.eye(n_modes)
        max_error = np.max(np.abs(inner_products - identity))

        # Detailed error metrics
        diagonal_errors = np.abs(np.diag(inner_products) - 1.0)
        max_diag_error = np.max(diagonal_errors)

        off_diagonal = inner_products.copy()
        np.fill_diagonal(off_diagonal, 0)
        max_off_diag = np.max(np.abs(off_diagonal))

        errors = {
            "max_error": max_error,
            "max_diagonal_error": max_diag_error,
            "max_off_diagonal": max_off_diag,
            "tolerance": tolerance,
        }

        # Determine if orthonormal
        is_orthonormal = max_error < tolerance

        if verbose:
            print(f"  Number of modes: {n_modes}")
            print(f"  Maximum deviation from identity: {max_error:.2e}")
            print(f"  Tolerance: {tolerance:.0e}")
            print(f"  Maximum diagonal error: {max_diag_error:.2e}")
            print(f"  Maximum off-diagonal: {max_off_diag:.2e}")
            print(
                f"  Result: {'✓ PASS (orthonormal)' if is_orthonormal else '✗ FAIL (not orthonormal)'}"
            )

        return is_orthonormal, max_error, errors

    def verify_eigendisplacement_orthonormality(
        self,
        q_index: int,
        tolerance: float = 1e-8,
        verbose: bool = False,
    ) -> Tuple[bool, float, Dict[str, Any]]:
        """
        Verify orthonormality of eigendisplacements using mass-weighted inner product.

        This method checks if the eigendisplacements at a specific q-point satisfy
        the orthonormality condition with respect to the mass-weighted inner product:

            <u_i|u_j>_M = δ_ij

        where u_i are the eigendisplacements obtained from get_eigen_displacement(),
        and <·|·>_M is the mass-weighted inner product. This forms the basis for
        verifying that the eigendisplacement matrix U satisfies U†MU = I.

        The verification computes the full orthonormality matrix:
            O_ij = <u_i|u_j>_M = Σ_k m_k * u_i,k* · u_j,k

        and checks that this matrix is approximately the identity matrix.

        IMPORTANT THEORETICAL NOTE:
        Eigendisplacements are mass-weighted transformations of raw eigenvectors:
        u_i = e_i / sqrt(M) (correct mass-weighting formula)

        With proper mass-weighting and normalization:
        - Raw eigenvectors are orthonormal under standard inner product: <e_i|e_j> = δ_ij ✓
        - Eigendisplacements ARE orthonormal under mass-weighted inner product: <u_i|u_j>_M = δ_ij ✓

        This method should typically return True for correct implementations.
        If it returns False, this suggests a bug in the mass-weighting or normalization.

        Parameters
        ----------
        q_index : int
            Index of the q-point to check
        tolerance : float, optional
            Numerical tolerance for orthonormality verification. Default is 1e-8
        verbose : bool, optional
            If True, print detailed information about the verification

        Returns
        -------
        Tuple[bool, float, Dict[str, Any]]
            - bool: True if eigendisplacements are orthonormal, False otherwise
            - float: Maximum deviation from identity matrix
            - Dict: Detailed analysis including orthonormality matrix and error metrics

        Raises
        ------
        ValueError
            If q_index is out of range

        Notes
        -----
        This method is specifically designed for eigendisplacements (mass-weighted),
        which differ from the raw eigenvectors. The eigendisplacements are obtained
        via get_eigen_displacement() and include proper mass-weighting and gauge
        transformations.

        For Gamma point calculations, this provides the most meaningful orthonormality
        test since eigendisplacements are real-valued and physically interpretable.

        Typical behavior: This method will return False (non-orthonormal) for most
        physical systems, which confirms correct mass-weighting implementation.

        Examples
        --------
        >>> modes = PhononModes(...)
        >>> is_orthonormal, max_error, details = modes.verify_eigendisplacement_orthonormality(0)
        >>> # Expected: is_orthonormal=False for most physical systems
        >>> print(f"Orthonormal: {is_orthonormal}, Max error: {max_error:.2e}")
        >>> print(f"Orthonormality matrix shape: {details['orthonormality_matrix'].shape}")
        """
        if q_index < 0 or q_index >= self.n_qpoints:
            raise ValueError(
                f"q_index {q_index} out of range [0, {self.n_qpoints - 1}]"
            )

        if verbose:
            print(
                f"\nVerifying eigendisplacement orthonormality at q-point index: {q_index}"
            )
            qpoint = self.qpoints[q_index]
            print(f"Q-point: ({qpoint[0]:.6f}, {qpoint[1]:.6f}, {qpoint[2]:.6f})")

        n_modes = self.n_modes

        # Get all eigendisplacements at the specified q-point
        eigendisplacements = []
        for mode_idx in range(n_modes):
            displacement = self.get_eigen_displacement(
                q_index=q_index, mode_index=mode_idx, normalize=True
            )
            eigendisplacements.append(displacement)

        if verbose:
            print(
                f"Computing orthonormality matrix for {n_modes} eigendisplacements..."
            )

        # Calculate orthonormality matrix: O_ij = <u_i|u_j>_M
        orthonormality_matrix = np.zeros((n_modes, n_modes), dtype=complex)

        for i in range(n_modes):
            for j in range(n_modes):
                # Use mass-weighted projection for orthonormality check
                projection = self.mass_weighted_projection(
                    eigendisplacements[i], eigendisplacements[j]
                )
                orthonormality_matrix[i, j] = projection

        # Check deviation from identity matrix
        identity = np.eye(n_modes)
        deviation_matrix = orthonormality_matrix - identity
        max_error = np.max(np.abs(deviation_matrix))

        # Detailed error analysis
        diagonal_elements = np.diag(orthonormality_matrix)
        diagonal_errors = np.abs(diagonal_elements - 1.0)
        max_diagonal_error = np.max(diagonal_errors)

        # Off-diagonal elements should be zero
        off_diagonal = orthonormality_matrix.copy()
        np.fill_diagonal(off_diagonal, 0)
        max_off_diagonal = np.max(np.abs(off_diagonal))

        # Determine if orthonormal within tolerance
        is_orthonormal = max_error < tolerance

        # Prepare detailed results
        details = {
            "orthonormality_matrix": orthonormality_matrix,
            "deviation_matrix": deviation_matrix,
            "max_error": max_error,
            "max_diagonal_error": max_diagonal_error,
            "max_off_diagonal": max_off_diagonal,
            "diagonal_elements": diagonal_elements,
            "diagonal_errors": diagonal_errors,
            "tolerance": tolerance,
            "n_modes": n_modes,
            "q_index": q_index,
            "qpoint": self.qpoints[q_index].copy(),
        }

        if verbose:
            print(f"  Number of modes: {n_modes}")
            print(f"  Maximum deviation from identity: {max_error:.2e}")
            print(f"  Tolerance: {tolerance:.0e}")
            print(f"  Maximum diagonal error: {max_diagonal_error:.2e}")
            print(f"  Maximum off-diagonal: {max_off_diagonal:.2e}")
            print(
                f"  Diagonal elements range: [{np.min(np.real(diagonal_elements)):.6f}, {np.max(np.real(diagonal_elements)):.6f}]"
            )

            # Check for any problematic modes
            problematic_modes = np.where(diagonal_errors > tolerance)[0]
            if len(problematic_modes) > 0:
                print(f"  Modes with diagonal errors > tolerance: {problematic_modes}")

            result_msg = (
                "✓ PASS (orthonormal)" if is_orthonormal else "✗ FAIL (not orthonormal)"
            )
            print(f"  Result: {result_msg}")

        return is_orthonormal, max_error, details

    def project_eigenvectors(self, q_index: int, unit_vector: np.ndarray) -> np.ndarray:
        """
        Project all eigenvectors at a specified q-point onto a user-supplied unit vector.

        Args:
            q_index: Index of the q-point
            unit_vector: Unit vector of matching dimension (n_atoms*3,)

        Returns:
            np.ndarray: Array of projection values (one per eigenvector)
        """
        if q_index < 0 or q_index >= self.n_qpoints:
            raise ValueError(
                f"q_index {q_index} out of range [0, {self.n_qpoints - 1}]"
            )
        eigenvectors_q = self.eigenvectors[q_index]  # shape: (n_modes, n_atoms*3)
        if unit_vector.shape[0] != eigenvectors_q.shape[1]:
            raise ValueError(
                f"unit_vector must have shape ({eigenvectors_q.shape[1]},)"
            )
        # Ensure unit_vector is normalized
        norm = np.linalg.norm(unit_vector)
        if not np.isclose(norm, 1.0, atol=1e-8):
            raise ValueError("unit_vector must be normalized (norm = 1)")
        # Compute projections
        projections = np.dot(eigenvectors_q, unit_vector)
        return projections

    def verify_completeness(
        self, q_index: int, unit_vector: np.ndarray, tolerance: float = 1e-8
    ) -> Tuple[bool, float]:
        """
        Verify that the sum of squared projections of all eigenvectors at a q-point onto a unit vector is 1 (within numerical tolerance).

        Args:
            q_index: Index of the q-point
            unit_vector: Unit vector of matching dimension (n_atoms*3,)
            tolerance: Numerical tolerance for completeness check

        Returns:
            (bool, float): Tuple of (is_complete, sum_of_squares)
        """
        projections = self.project_eigenvectors(q_index, unit_vector)
        sum_of_squares = np.sum(np.abs(projections) ** 2)
        is_complete = np.isclose(sum_of_squares, 1.0, atol=tolerance)
        return is_complete, sum_of_squares

    def print_projection_table(
        self,
        displacement: np.ndarray,
        supercell_matrix: np.ndarray,
        displacement_masses: Optional[np.ndarray] = None,
        normalize_displacement: bool = True,
        max_modes_per_qpoint: Optional[int] = None,
        show_frequencies: bool = True,
        precision: int = 6,
    ) -> Dict[str, Union[int, float, Dict[int, float]]]:
        """
        Print a detailed table showing projections of a displacement onto all commensurate q-point modes.

        This method provides comprehensive analysis of how a given displacement decomposes
        across all phonon modes from commensurate q-points, with per-q-point and total sums.

        Args:
            displacement: Displacement vector (n_supercell_atoms, 3) or (n_supercell_atoms*3,)
            supercell_matrix: 3x3 supercell transformation matrix
            displacement_masses: Optional masses for the displacement. If None, will construct
                                from primitive cell masses tiled for supercell.
            normalize_displacement: If True, normalize displacement to unit mass-weighted norm
            max_modes_per_qpoint: Maximum number of modes to show per q-point (None = all)
            show_frequencies: If True, include frequency information in the table
            precision: Number of decimal places for numerical output

        Returns:
            dict: Dictionary containing summary statistics:
                  - 'qpoint_sums': dict mapping q-point indices to their sum of |projection|²
                  - 'total_sum': total sum of all |projection|²
                  - 'n_qpoints': number of commensurate q-points
                  - 'n_total_modes': total number of modes
        """
        # Validate and reshape displacement
        displacement = np.asarray(displacement)
        if displacement.ndim == 2:
            displacement = displacement.reshape(-1)

        n_supercell_atoms = len(displacement) // 3

        # Get supercell masses
        if displacement_masses is None:
            n_primitive_cells = abs(np.linalg.det(supercell_matrix))
            displacement_masses = np.tile(self.atomic_masses, int(n_primitive_cells))

        # Normalize displacement if requested
        if normalize_displacement:
            displacement_norm = self.mass_weighted_norm(
                displacement.reshape(-1, 3), displacement_masses
            )
            if displacement_norm > 1e-12:
                displacement = displacement / displacement_norm
                print("Input displacement normalized to unit mass-weighted norm")
            else:
                print("Warning: Input displacement has zero norm!")

        # Get commensurate q-points and their displacements
        try:
            all_commensurate_displacements = (
                self.generate_all_commensurate_displacements(
                    supercell_matrix, amplitude=1.0
                )
            )
        except Exception as e:
            print(f"Error generating commensurate displacements: {e}")
            return {}

        if len(all_commensurate_displacements) == 0:
            print("No commensurate q-points found!")
            return {}

        # Print header
        print("\n" + "=" * 80)
        print("PROJECTION TABLE: Displacement onto Commensurate Q-point Modes")
        print("=" * 80)
        print(f"Supercell matrix: {supercell_matrix.tolist()}")
        print(f"Supercell atoms: {n_supercell_atoms}")
        print(
            f"Displacement norm: {self.mass_weighted_norm(displacement.reshape(-1, 3), displacement_masses):.{precision}f}"
        )
        print(f"Commensurate q-points: {len(all_commensurate_displacements)}")

        # Calculate projections and build table
        qpoint_sums = {}
        total_sum = 0.0
        n_total_modes = 0

        for q_index, mode_displacements in all_commensurate_displacements.items():
            qpoint = self.qpoints[q_index]
            n_modes = mode_displacements.shape[0]
            n_total_modes += n_modes

            # Limit modes if requested
            modes_to_show = (
                min(n_modes, max_modes_per_qpoint) if max_modes_per_qpoint else n_modes
            )

            print(
                f"\n--- Q-point {q_index}: [{qpoint[0]:.3f}, {qpoint[1]:.3f}, {qpoint[2]:.3f}] ---"
            )
            print(f"Modes: {n_modes} total, showing {modes_to_show}")

            if show_frequencies:
                header = f"{'Mode':<6} {'Freq (THz)':<12} {'|Projection|²':<15} {'Re(Proj)':<12} {'Im(Proj)':<12}"
            else:
                header = f"{'Mode':<6} {'|Projection|²':<15} {'Re(Proj)':<12} {'Im(Proj)':<12}"
            print(header)
            print("-" * len(header))

            qpoint_sum = 0.0

            for mode_idx in range(modes_to_show):
                # Calculate projection
                mode_displacement = mode_displacements[mode_idx].reshape(-1)
                projection = self.mass_weighted_projection(
                    displacement, mode_displacement, displacement_masses
                )
                projection_magnitude_sq = abs(projection) ** 2
                qpoint_sum += projection_magnitude_sq

                # Format output
                if show_frequencies:
                    freq = self.frequencies[q_index, mode_idx]
                    row = f"{mode_idx:<6} {freq:<12.{precision}f} {projection_magnitude_sq:<15.{precision}f} "
                else:
                    row = f"{mode_idx:<6} {projection_magnitude_sq:<15.{precision}f} "

                row += f"{projection.real:<12.{precision}f} {projection.imag:<12.{precision}f}"
                print(row)

            if modes_to_show < n_modes:
                print(f"... ({n_modes - modes_to_show} more modes)")
                # Calculate remaining projections for sum
                for mode_idx in range(modes_to_show, n_modes):
                    mode_displacement = mode_displacements[mode_idx].reshape(-1)
                    projection = self.mass_weighted_projection(
                        displacement, mode_displacement, displacement_masses
                    )
                    qpoint_sum += abs(projection) ** 2

            print(f"Q-point {q_index} sum of |projection|²: {qpoint_sum:.{precision}f}")
            qpoint_sums[q_index] = qpoint_sum
            total_sum += qpoint_sum

        # Print summary
        print("\n" + "=" * 80)
        print("SUMMARY")
        print("=" * 80)
        print(f"{'Q-point':<10} {'Sum |Proj|²':<15} {'Q-vector':<20}")
        print("-" * 50)

        for q_index in sorted(qpoint_sums.keys()):
            qpoint = self.qpoints[q_index]
            qvec_str = f"[{qpoint[0]:.3f}, {qpoint[1]:.3f}, {qpoint[2]:.3f}]"
            print(
                f"{q_index:<10} {qpoint_sums[q_index]:<15.{precision}f} {qvec_str:<20}"
            )

        print("-" * 50)
        print(f"{'TOTAL':<10} {total_sum:<15.{precision}f}")
        print(
            f"\nCompleteness check: {total_sum:.{precision}f} (expect ≈ 1.0 for normalized basis)"
        )

        if abs(total_sum - 1.0) < 1e-3:
            print("✅ Excellent completeness - modes form proper orthonormal basis")
        elif abs(total_sum - 1.0) < 1e-2:
            print(
                "✓ Good completeness - small deviations likely due to numerical precision"
            )
        else:
            print(
                "⚠️  Large deviation from completeness - check q-point coverage or normalization"
            )

        print("=" * 80)

        return {
            "qpoint_sums": qpoint_sums,
            "total_sum": total_sum,
            "n_qpoints": len(all_commensurate_displacements),
            "n_total_modes": n_total_modes,
        }

    def print_eigenvectors(
        self,
        q_index: int = 0,
        mode_indices: Optional[List[int]] = None,
        max_components: int = 10,
        precision: int = 6,
        show_magnitudes: bool = True,
    ) -> None:
        """
        Print eigenvectors for specified q-point and modes.

        This method displays the raw eigenvector components from the phonon calculation.
        These are unit-normalized (Euclidean norm = 1) but not mass-normalized.

        Args:
            q_index: Index of the q-point (default: 0)
            mode_indices: List of mode indices to print. If None, prints all modes
            max_components: Maximum number of components to print per mode
            precision: Number of decimal places for printing
            show_magnitudes: Whether to show magnitude of each component

        Example:
            >>> modes.print_eigenvectors(q_index=0, mode_indices=[0, 1, 2])
        """
        if q_index < 0 or q_index >= self.n_qpoints:
            raise ValueError(
                f"q_index {q_index} out of range [0, {self.n_qpoints - 1}]"
            )

        if mode_indices is None:
            mode_indices = list(range(self.n_modes))
        else:
            for mode_idx in mode_indices:
                if mode_idx < 0 or mode_idx >= self.n_modes:
                    raise ValueError(
                        f"mode_index {mode_idx} out of range [0, {self.n_modes - 1}]"
                    )

        qpoint = self.qpoints[q_index]
        print(
            f"\nEigenvectors at q-point {q_index}: ({qpoint[0]:.3f}, {qpoint[1]:.3f}, {qpoint[2]:.3f})"
        )
        print("=" * 80)

        for mode_idx in mode_indices:
            frequency, eigenvector = self.get_mode(q_index, mode_idx)

            # Determine mode type
            if abs(frequency) < 1e-6:
                mode_type = "acoustic"
            elif frequency < 0:
                mode_type = "imaginary"
            else:
                mode_type = "optical"

            print(f"\nMode {mode_idx:2d} ({mode_type}): {frequency:8.3f} THz")
            print(f"  Eigenvector shape: {eigenvector.shape}")

            # Show components
            n_components = len(eigenvector)
            n_show = min(n_components, max_components)

            print("  Components (real, imag):")
            for i in range(n_show):
                real_part = np.real(eigenvector[i])
                imag_part = np.imag(eigenvector[i])

                if show_magnitudes:
                    magnitude = np.abs(eigenvector[i])
                    print(
                        f"    [{i:2d}]: {real_part: .{precision}f} {imag_part:+.{precision}f}j  |{eigenvector[i]:.{precision}f}| = {magnitude:.{precision}f}"
                    )
                else:
                    print(
                        f"    [{i:2d}]: {real_part: .{precision}f} {imag_part:+.{precision}f}j"
                    )

            if n_components > max_components:
                print(f"    ... ({n_components - max_components} more components)")

            # Show Euclidean norm
            euclidean_norm = np.linalg.norm(eigenvector)
            print(f"  Euclidean norm: {euclidean_norm:.{precision}f}")

    def print_eigen_displacements(
        self,
        q_index: int = 0,
        mode_indices: Optional[List[int]] = None,
        supercell_matrix: Optional[np.ndarray] = None,
        amplitude: float = 0.1,
        max_atoms: int = 10,
        precision: int = 6,
        show_masses: bool = True,
    ) -> None:
        """
        Print eigen displacements for specified q-point and modes.

        This method displays the actual atomic displacement patterns that result from
        the phonon eigenvectors when applied to a supercell structure.

        Args:
            q_index: Index of the q-point (default: 0)
            mode_indices: List of mode indices to print. If None, prints all modes
            supercell_matrix: 3x3 supercell transformation matrix. If None, uses 1x1x1
            amplitude: Displacement amplitude in Angstroms (default: 0.1)
            max_atoms: Maximum number of atoms to show per mode
            precision: Number of decimal places for printing
            show_masses: Whether to show atomic masses

        Example:
            >>> modes.print_eigen_displacements(q_index=0, mode_indices=[0, 1, 2],
            ...                               supercell_matrix=np.eye(3)*2)
        """
        if q_index < 0 or q_index >= self.n_qpoints:
            raise ValueError(
                f"q_index {q_index} out of range [0, {self.n_qpoints - 1}]"
            )

        if mode_indices is None:
            mode_indices = list(range(self.n_modes))
        else:
            for mode_idx in mode_indices:
                if mode_idx < 0 or mode_idx >= self.n_modes:
                    raise ValueError(
                        f"mode_index {mode_idx} out of range [0, {self.n_modes - 1}]"
                    )

        if supercell_matrix is None:
            supercell_matrix = np.eye(3)

        # Generate supercell to get masses
        from phonproj.core.supercell import generate_supercell

        supercell = generate_supercell(self.primitive_cell, supercell_matrix)
        supercell_masses = supercell.get_masses()

        qpoint = self.qpoints[q_index]
        det = int(round(np.linalg.det(supercell_matrix)))

        print(
            f"\nEigen Displacements at q-point {q_index}: ({qpoint[0]:.3f}, {qpoint[1]:.3f}, {qpoint[2]:.3f})"
        )
        print(f"Supercell matrix:\n{supercell_matrix}")
        print(f"Supercell size: {det}x primitive cell ({len(supercell)} atoms)")
        print(f"Amplitude: {amplitude} Å")
        print("=" * 80)

        for mode_idx in mode_indices:
            frequency, eigenvector = self.get_mode(q_index, mode_idx)

            # Determine mode type
            if abs(frequency) < 1e-6:
                mode_type = "acoustic"
            elif frequency < 0:
                mode_type = "imaginary"
            else:
                mode_type = "optical"

            print(f"\nMode {mode_idx:2d} ({mode_type}): {frequency:8.3f} THz")

            # Generate eigen displacement
            eigen_displacement = self.generate_mode_displacement(
                q_index=q_index,
                mode_index=mode_idx,
                supercell_matrix=supercell_matrix,
                amplitude=amplitude,
                normalize=False,
            )

            print(f"  Displacement shape: {eigen_displacement.shape}")

            # Show displacements for atoms
            n_atoms = len(eigen_displacement)
            n_show = min(n_atoms, max_atoms)

            print("  Atomic displacements (x, y, z) in Å:")
            print(
                "  Atom |   Mass    |         Δx            Δy            Δz        |   |Δ|"
            )
            print(
                "  -----|-----------|-------------------------------------------|---------"
            )

            for i in range(n_show):
                disp = eigen_displacement[i]
                disp_magnitude = np.linalg.norm(disp)

                if show_masses:
                    mass = supercell_masses[i]
                    print(
                        f"  {i:4d} | {mass:8.3f} | {disp[0]: .{precision}f} {disp[1]: .{precision}f} {disp[2]: .{precision}f} | {disp_magnitude:.{precision}f}"
                    )
                else:
                    print(
                        f"  {i:4d} |         -- | {disp[0]: .{precision}f} {disp[1]: .{precision}f} {disp[2]: .{precision}f} | {disp_magnitude:.{precision}f}"
                    )

            if n_atoms > max_atoms:
                print(f"       ... ({n_atoms - max_atoms} more atoms)")

            # Show mass-weighted norm
            mass_norm = self.mass_weighted_norm(eigen_displacement, supercell_masses)
            max_displacement = np.max(np.abs(eigen_displacement))
            print(f"  Max displacement: {max_displacement:.{precision}f} Å")
            print(f"  Mass-weighted norm: {mass_norm:.{precision}f}")

    def generate_eigen_displacement_phonopy(
        self,
        q_index: int,
        mode_index: int,
        supercell_matrix: np.ndarray,
        amplitude: float = 0.1,
        return_phonopy_objects: bool = False,
    ) -> Union[np.ndarray, Tuple[np.ndarray, Any]]:
        """
        Generate eigen displacement using Phonopy's get_modulations_and_supercell method.

        This method provides an alternative approach that uses Phonopy's native API
        for computing phonon displacements, which may be useful for compatibility
        and comparison with Phonopy's internal calculations.

        The method uses phonopy.get_modulations_and_supercell() which handles:
        - Modulation of atomic positions based on phonon eigenvectors
        - Supercell generation with proper phase factors
        - Gauge transformations and mass-weighting

        Args:
            q_index: Index of the q-point
            mode_index: Index of the phonon mode
            supercell_matrix: 3x3 supercell transformation matrix
            amplitude: Displacement amplitude in Angstroms (default: 0.1)
            factor: Additional scaling factor (default: 1.0)
            return_phonopy_objects: If True, returns (displacements, phonopy_supercell, phonopy_modulations)

        Returns:
            numpy.ndarray: Real displacement vectors of shape (n_supercell_atoms, 3)
            If return_phonopy_objects=True: tuple of (displacements, phonopy_supercell, phonopy_modulations)

        Raises:
            ImportError: If phonopy is not available
            ValueError: If indices are out of range or supercell_matrix is invalid

        Example:
            >>> import numpy as np
            >>> # Generate displacements using Phonopy's method
            >>> displacements = modes.generate_eigen_displacement_phonopy(
            ...     q_index=0, mode_index=3, supercell_matrix=np.eye(3)*2,
            ...     amplitude=0.05
            ... )
            >>> print(f"Generated {len(displacements)} displacement vectors")
        """
        try:
            import phonopy
        except ImportError as e:
            raise ImportError(
                "phonopy is required for generate_eigen_displacement_phonopy(). "
                "Install it with: pip install phonopy"
            ) from e

        # Validate inputs
        if q_index < 0 or q_index >= self.n_qpoints:
            raise ValueError(
                f"q_index {q_index} out of range [0, {self.n_qpoints - 1}]"
            )

        if mode_index < 0 or mode_index >= self.n_modes:
            raise ValueError(
                f"mode_index {mode_index} out of range [0, {self.n_modes - 1}]"
            )

        supercell_matrix = np.asarray(supercell_matrix, dtype=int)
        if supercell_matrix.shape != (3, 3):
            raise ValueError("supercell_matrix must be a 3x3 integer matrix")

        # Get qpoint and mode frequency
        qpoint = self.qpoints[q_index]
        frequency = self.frequencies[q_index, mode_index]

        # Create a temporary Phonopy object for this calculation
        # Convert ASE structure to Phonopy format
        phonopy_unitcell = self._convert_ase_to_phonopy_cell()

        # Set up eigenvector for this mode
        # Phonopy expects eigenvectors in shape (3*n_atoms, n_modes)
        all_eigenvectors = np.zeros((3 * self.n_atoms, 1), dtype=complex)
        all_eigenvectors[:, 0] = self.eigenvectors[q_index, mode_index]

        # Set frequencies for all modes (we only care about one)
        np.array([frequency])

        # Create a temporary Phonopy object
        phonon = phonopy.Phonopy(
            phonopy_unitcell, supercell_matrix, primitive_matrix=None, nac_params=None
        )

        # Get the mode displacements using Phonopy's method
        # This is the key difference - using Phonopy's get_modulations_and_supercell
        try:
            # Need to set up the phonon object with force constants and calculate the eigenvectors
            # For now, we'll use a simplified approach that leverages the existing eigenvectors

            # Define phonon mode for modulation
            # Format: [q-point, band index, amplitude, phase]
            phonon_modes = [[qpoint.tolist(), mode_index, amplitude, 0.0]]

            # Run modulations to generate the displacements
            phonon.run_modulations(
                dimension=supercell_matrix,
                phonon_modes=phonon_modes,
            )

            # Get modulations and supercell
            phonopy_modulations, phonopy_supercell = (
                phonon.get_modulations_and_supercell()
            )

            # Extract displacements from the modulations
            # The modulations array contains the displacement vectors for all atoms
            displacements = phonopy_modulations

            if return_phonopy_objects:
                return displacements, (phonopy_supercell, phonopy_modulations)
            else:
                return displacements

        except Exception as e:
            # If phonopy method fails, provide a helpful error message
            raise RuntimeError(
                f"Failed to generate eigen displacements using Phonopy method: {e}\n"
                f"This may be due to Phonopy version compatibility or API changes.\n"
                f"Falling back to the standard method or checking your Phonopy installation."
            ) from e

    def _convert_ase_to_phonopy_cell(self) -> Any:
        """Convert ASE Atoms to Phonopy unit cell format."""
        import phonopy.structure.atoms as phonopy_atoms

        positions = self.primitive_cell.get_positions()
        cell = self.primitive_cell.get_cell()
        numbers = self.primitive_cell.get_atomic_numbers()
        masses = self.atomic_masses

        # Create Phonopy unit cell using PhonopyAtoms
        phonopy_unitcell = phonopy_atoms.PhonopyAtoms(
            numbers=numbers, positions=positions, cell=cell, masses=masses, pbc=True
        )

        return phonopy_unitcell

    def _get_phonopy_primitive(self) -> Any:
        """Get the primitive cell as a PhonopyAtoms object."""
        return self._convert_ase_to_phonopy_cell()

    def _get_supercell_original_positions(
        self, phonopy_supercell, supercell_matrix
    ) -> np.ndarray:
        """Get original positions for a supercell."""
        # Create the supercell without any modulation
        supercell = make_supercell(self.primitive_cell, supercell_matrix)
        return supercell.get_positions()

    def calculate_band_structure(
        self,
        path: Optional[str] = None,
        npoints: int = 100,
        units: str = "cm-1",  # Changed default to cm-1 for better visualization
    ) -> "PhononBand":
        """
        Calculate phonon band structure along a high-symmetry k-path.

        This method creates a PhononBand object by generating a k-path and
        calculating frequencies along that path.

        Parameters
        ----------
        path : str, optional
            User-specified k-path (e.g., 'GXMGR', 'Γ-X-M-Γ-R').
            If None, automatically generates path based on crystal symmetry.
        npoints : int, optional
            Number of k-points along the path. Default is 100.
        units : str, optional
            Frequency units ('THz', 'cm-1', 'meV'). Default is 'THz'.

        Returns
        -------
        PhononBand
            PhononBand object with calculated band structure

        Examples
        --------
        >>> band = modes.calculate_band_structure()  # Automatic path
        >>> band = modes.calculate_band_structure(path='GXMGR')  # Manual path
        >>> band.plot()  # Plot the band structure
        """
        from phonproj.band_structure import PhononBand

        return PhononBand.from_modes_with_path(
            modes=self, path=path, npoints=npoints, units=units
        )

    def find_nearest_atoms(
        self, target_structure: Atoms, symprec: float = 1e-5
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Find nearest atoms in target structure for each atom in this PhononModes.

        Args:
            target_structure: Target structure to search for nearest atoms
            symprec: Symmetry precision

        Returns:
            Tuple of (indices, distances): Results from nearest atom search
        """
        from phonproj.core.structure_analysis import find_nearest_atoms

        indices, distances, _ = find_nearest_atoms(
            self.primitive_cell,
            target_structure,
            cell1=self.primitive_cell.get_cell(),
            cell2=target_structure.get_cell(),
            symprec=symprec,
        )
        return indices, distances

    def create_atom_mapping(
        self, target_structure: Atoms, method: str = "distance", max_cost: float = 1e-3
    ) -> Tuple[np.ndarray, float]:
        """
        Create atom correspondence between this PhononModes and target structure.

        Args:
            target_structure: Target structure to map to
            method: Matching method ('distance' only supported currently)
            max_cost: Maximum allowed mapping cost

        Returns:
            Tuple of (mapping, cost): Atom mapping and total cost
        """
        from phonproj.core.structure_analysis import create_atom_mapping

        return create_atom_mapping(
            self.primitive_cell, target_structure, method=method, max_cost=max_cost
        )

    def reorder_structure_by_mapping(self, mapping: np.ndarray) -> Atoms:
        """
        Reorder this PhononModes structure according to atom mapping.

        Args:
            mapping: Array where mapping[i] = j means new position i should have
                     atom that was originally at position j

        Returns:
            Reordered Atoms object
        """
        from phonproj.core.structure_analysis import reorder_structure_by_mapping

        return reorder_structure_by_mapping(self.primitive_cell, mapping)

    def generate_supercell(
        self, supercell_matrix: np.ndarray, symprec: float = 1e-5
    ) -> Atoms:
        """
        Generate a supercell from the primitive cell using the phonopy algorithm.

        This is a convenience method that calls the standalone generate_supercell function
        using this PhononModes object's primitive cell.

        Args:
            supercell_matrix: 3x3 integer transformation matrix
            symprec: Symmetry precision for operations (default: 1e-5)

        Returns:
            ASE Atoms object representing the supercell

        Raises:
            ValueError: If supercell_matrix is not a valid 3x3 integer matrix

        Examples:
            >>> import numpy as np
            >>>
            >>> # Create 2x2x2 supercell
            >>> supercell_matrix = np.eye(3) * 2
            >>> supercell = modes.generate_supercell(supercell_matrix)
            >>> print(f"Supercell has {len(supercell)} atoms")
        """
        from phonproj.core.supercell import generate_supercell

        return generate_supercell(
            primitive_cell=self.primitive_cell,
            supercell_matrix=supercell_matrix,
            symprec=symprec,
        )

    def generate_mode_displacement(
        self,
        q_index: int,
        mode_index: int,
        supercell_matrix: np.ndarray,
        amplitude: float = 0.1,
        argument: float = 0.0,
        mod_func: Optional[Callable] = None,
        use_isotropy_amplitude: bool = True,
        normalize: bool = False,
    ) -> np.ndarray:
        """
        Generate displacement patterns for a phonon mode in a supercell.

        This is a convenience method that calls the standalone generate_mode_displacement
        function using this PhononModes object's data. By default, displacements are not
        normalized and include a 1/sqrt(N_cells) factor where N_cells is the number of
        primitive cells in the supercell.

        Args:
            q_index: Index of the q-point
            mode_index: Index of the phonon mode
            supercell_matrix: 3x3 supercell transformation matrix
            amplitude: Displacement amplitude in Angstroms (default: 0.1)
            argument: Phase argument in radians (default: 0.0)
            mod_func: Optional modulation function for spatial variation
            use_isotropy_amplitude: Whether to use isotropy amplitude (default: True)
            normalize: Whether to normalize displacements (default: False)

        Returns:
            numpy.ndarray: Complex displacement vectors of shape (n_supercell_atoms, 3)

        Raises:
            ValueError: If indices are out of range or q-point is not commensurate

        Examples:
            >>> import numpy as np
            >>>
            >>> # Generate displacements for mode 3 at q-point 0
            >>> supercell_matrix = np.eye(3) * 2
            >>> displacements = modes.generate_mode_displacement(
            ...     q_index=0, mode_index=3,
            ...     supercell_matrix=supercell_matrix, amplitude=0.05
            ... )
            >>> print(f"Generated {len(displacements)} displacement vectors")
        """
        from phonproj.core.supercell import generate_mode_displacement

        return generate_mode_displacement(
            modes=self,
            q_index=q_index,
            mode_index=mode_index,
            supercell_matrix=supercell_matrix,
            amplitude=amplitude,
            argument=argument,
            mod_func=mod_func,
            use_isotropy_amplitude=use_isotropy_amplitude,
            normalize=normalize,
        )

    def project_displacement_with_phase_scan(
        self,
        q_index: int,
        mode_index: int,
        target_displacement: np.ndarray,
        supercell_matrix: np.ndarray,
        n_phases: int = 360,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Scan projection coefficients as a function of phase angle for a specific mode.

        This is a convenience method that creates a supercell and calls the
        standalone project_displacement_with_phase_scan function.

        Args:
            q_index: Index of the q-point
            mode_index: Index of the phonon mode
            target_displacement: Target displacement pattern (n_target_atoms, 3)
            supercell_matrix: 3x3 supercell transformation matrix
            n_phases: Number of phase points to sample (default: 360)

        Returns:
            Tuple of (phases, coefficients):
                phases: Array of phase angles in radians from 0 to 2π
                coefficients: Array of projection coefficients at each phase angle

        Example:
            >>> phases, coeffs = modes.project_displacement_with_phase_scan(
            ...     q_index=0, mode_index=3, target_disp=target_disp,
            ...     supercell_matrix=supercell_matrix, n_phases=180
            ... )
            >>> print(f"Generated {len(phases)} phase points")
        """
        from phonproj.core.structure_analysis import (
            project_displacement_with_phase_scan,
        )

        # Create source supercell
        source_supercell = self.generate_displaced_supercell(
            q_index=q_index,
            mode_index=mode_index,
            supercell_matrix=supercell_matrix,
            amplitude=0.0,  # Zero amplitude to get undisplaced supercell
            return_displacements=False,
        )
        assert isinstance(
            source_supercell, Atoms
        ), "Expected Atoms object when return_displacements=False"

        # Create target supercell (same as source for projection analysis)
        target_supercell = self.generate_displaced_supercell(
            q_index=q_index,
            mode_index=mode_index,
            supercell_matrix=supercell_matrix,
            amplitude=0.0,  # Zero amplitude to get undisplaced supercell
            return_displacements=False,
        )
        assert isinstance(
            target_supercell, Atoms
        ), "Expected Atoms object when return_displacements=False"

        return project_displacement_with_phase_scan(
            self,
            target_displacement,
            source_supercell,
            target_supercell,
            supercell_matrix,
            atom_mapping=None,
            n_phases=n_phases,
        )

    def find_optimal_phase(
        self,
        q_index: int,
        mode_index: int,
        target_displacement: np.ndarray,
        supercell_matrix: np.ndarray,
        n_phases: int = 360,
    ) -> Tuple[float, float]:
        """
        Find the optimal phase angle that maximizes projection coefficient.

        Args:
            q_index: Index of the q-point
            mode_index: Index of the phonon mode
            target_displacement: Target displacement pattern (n_target_atoms, 3)
            supercell_matrix: 3x3 supercell transformation matrix
            n_phases: Number of phase points to sample (default: 360)

        Returns:
            Tuple of (max_coefficient, optimal_phase):
                max_coefficient: Maximum absolute projection coefficient
                optimal_phase: Phase angle in radians corresponding to maximum coefficient

        Example:
            >>> max_coeff, optimal_phase = modes.find_optimal_phase(
            ...     q_index=0, mode_index=3, target_disp=target_disp,
            ...     supercell_matrix=supercell_matrix
            ... )
            >>> print(f"Maximum: {max_coeff:.6f} at phase {optimal_phase:.3f} rad")
        """
        from phonproj.core.structure_analysis import find_maximum_projection

        # Get phase scan results
        phases, coefficients = self.project_displacement_with_phase_scan(
            q_index, mode_index, target_displacement, supercell_matrix, n_phases
        )

        # Find optimal phase
        return find_maximum_projection(phases, coefficients)

    def generate_displaced_supercell(
        self,
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
        """
        Generate a supercell with phonon mode displacements applied.

        This is a convenience method that calls the standalone generate_displaced_supercell
        function using this PhononModes object's data. By default, displacements are not
        normalized and include a 1/sqrt(N_cells) factor where N_cells is the number of
        primitive cells in the supercell.

        Args:
            q_index: Index of the q-point
            mode_index: Index of the phonon mode
            supercell_matrix: 3x3 supercell transformation matrix
            amplitude: Displacement amplitude in Angstroms (default: 0.1)
            argument: Phase argument in radians (default: 0.0)
            mod_func: Optional modulation function for spatial variation
            use_isotropy_amplitude: Whether to use isotropy amplitude (default: True)
            return_displacements: If True, also return displacement vectors
            normalize: Whether to normalize displacements (default: False)

        Returns:
            ASE Atoms: Displaced supercell structure
            If return_displacements=True: tuple of (displaced_supercell, displacements)

        Raises:
            ValueError: If indices are out of range or q-point is not commensurate

        Examples:
            >>> import numpy as np
            >>>
            >>> # Generate displaced supercell for mode 3 at q-point 0
            >>> supercell_matrix = np.eye(3) * 2
            >>> displaced_cell = modes.generate_displaced_supercell(
            ...     q_index=0, mode_index=3,
            ...     supercell_matrix=supercell_matrix, amplitude=0.05
            ... )
            >>> print(f"Displaced supercell has {len(displaced_cell)} atoms")
            >>>
            >>> # Get both structure and displacements
            >>> structure, disps = modes.generate_displaced_supercell(
            ...     q_index=0, mode_index=3,
            ...     supercell_matrix=supercell_matrix,
            ...     return_displacements=True
            ... )
        """
        from phonproj.core.supercell import generate_displaced_supercell

        return generate_displaced_supercell(
            modes=self,
            q_index=q_index,
            mode_index=mode_index,
            supercell_matrix=supercell_matrix,
            amplitude=amplitude,
            argument=argument,
            mod_func=mod_func,
            use_isotropy_amplitude=use_isotropy_amplitude,
            return_displacements=return_displacements,
            normalize=normalize,
        )

    @staticmethod
    def from_phonopy_yaml(yaml_file: str, qpoints: np.ndarray) -> "PhononModes":
        """
        Create PhononModes object from phonopy YAML file at specific q-points.
        """
        from pathlib import Path

        from phonproj.core.io import (
            _calculate_phonons_at_kpoints,
            create_phonopy_object,
            phonopy_to_ase,
        )

        # Create phonopy object from YAML file
        phonopy = create_phonopy_object(Path(yaml_file))

        # Calculate phonons at the specified q-points
        frequencies, eigenvectors = _calculate_phonons_at_kpoints(phonopy, qpoints)

        # Convert phonopy primitive cell to ASE Atoms
        primitive_cell = phonopy_to_ase(phonopy.primitive)

        # Create atomic masses
        atomic_masses = np.array(phonopy.primitive.masses)

        # Create and return PhononModes object
        modes = PhononModes(
            primitive_cell=primitive_cell,
            qpoints=qpoints,
            frequencies=frequencies,
            eigenvectors=eigenvectors,
            atomic_masses=atomic_masses,
            gauge="R",
        )

        # Store phonopy metadata for phonopy modulation API access
        modes._phonopy_yaml_path = str(yaml_file)
        modes._phonopy_directory = None  # type: ignore

        return modes

    @classmethod
    def from_phonopy_directory(
        cls,
        directory: str,
        qpoints: Optional[np.ndarray] = None,
        n_qpoints: int = 100,
        symprec: float = 1e-5,
        angle_tolerance: float = -1.0,
        **kwargs: Any,
    ) -> "PhononModes":
        """
        Create PhononModes object from a directory containing Phonopy files.
        """
        from pathlib import Path

        from phonproj.core.io import load_from_phonopy_files

        data = load_from_phonopy_files(Path(directory), **kwargs)
        phonopy = data["phonopy"]

        # Generate q-points if not provided
        if qpoints is None:
            n_qpoints_array = (
                np.array(n_qpoints)
                if hasattr(n_qpoints, "__iter__")
                else np.array([n_qpoints, n_qpoints, n_qpoints])
            )
            mesh_points = []
            for i in range(n_qpoints_array[0]):
                for j in range(n_qpoints_array[1]):
                    for k in range(n_qpoints_array[2]):
                        q = np.array(
                            [
                                i / n_qpoints_array[0],
                                j / n_qpoints_array[1],
                                k / n_qpoints_array[2],
                            ]
                        )
                        mesh_points.append(q)
            qpoints = np.array(mesh_points)

        # Calculate phonons at the specified q-points
        from phonproj.core.io import _calculate_phonons_at_kpoints, phonopy_to_ase

        frequencies, eigenvectors = _calculate_phonons_at_kpoints(phonopy, qpoints)

        # Convert phonopy primitive cell to ASE Atoms
        primitive_cell = phonopy_to_ase(phonopy.primitive)

        # Create atomic masses
        atomic_masses = np.array(phonopy.primitive.masses)

        # Create and return PhononModes object
        modes = PhononModes(
            primitive_cell=primitive_cell,
            qpoints=qpoints,
            frequencies=frequencies,
            eigenvectors=eigenvectors,
            atomic_masses=atomic_masses,
            gauge="R",
        )

        # Store phonopy metadata for phonopy modulation API access
        modes._phonopy_directory = str(directory)
        modes._phonopy_yaml_path = None  # type: ignore  # type: ignore

        return modes

    def decompose_displacement(
        self,
        displacement: np.ndarray,
        supercell_matrix: np.ndarray,
        normalize: bool = True,
        tolerance: float = 1e-12,
        print_table: bool = True,
        max_entries: int = 20,
        min_contribution: float = 1e-6,
    ) -> Tuple[List[dict], dict]:
        """
        Decompose an arbitrary displacement into contributions from all phonon modes.

        This is a convenience method that implements Step 9 complete mode decomposition,
        projecting any displacement vector onto all phonon modes from all commensurate
        q-points in the specified supercell.

        Args:
            displacement: Displacement pattern to decompose (n_atoms, 3)
            supercell_matrix: 3x3 supercell transformation matrix
            normalize: Whether to normalize displacement before decomposition
            tolerance: Numerical tolerance for completeness verification
            print_table: Whether to print formatted results table
            max_entries: Maximum table entries to display if printing
            min_contribution: Minimum squared coefficient to display in table

        Returns:
            Tuple of (projection_table, summary):
            - projection_table: List of dicts with projection data for each mode
            - summary: Dict with completeness verification and statistics

        Example:
            >>> # Decompose a random displacement
            >>> displacement = np.random.random((8, 3))  # 2x2x2 supercell
            >>> table, summary = modes.decompose_displacement(
            ...     displacement, np.eye(3)*2
            ... )
            >>> print(f"Completeness: {summary['is_complete']}")
        """
        from phonproj.core.structure_analysis import (
            decompose_displacement_to_modes,
            print_decomposition_table,
        )

        # Perform decomposition
        projection_table, summary = decompose_displacement_to_modes(
            displacement=displacement,
            phonon_modes=self,
            supercell_matrix=supercell_matrix,
            normalize=normalize,
            tolerance=tolerance,
        )

        # Print table if requested
        if print_table:
            print_decomposition_table(
                projection_table=projection_table,
                summary=summary,
                max_entries=max_entries,
                min_contribution=min_contribution,
            )

        return projection_table, summary

    def project_displacement_to_supercell(
        self,
        source_displacement: np.ndarray,
        source_supercell: Atoms,
        target_supercell: Atoms,
        normalize: bool = True,
        atom_mapping: Optional[np.ndarray] = None,
    ) -> float:
        """
        Project a displacement from source supercell onto target supercell.

        This is a convenience method that projects a displacement pattern from one
        supercell representation onto another supercell representation, handling
        atom ordering differences automatically.

        Args:
            source_displacement: Displacement pattern in source supercell (n_source_atoms, 3)
            source_supercell: Source supercell structure
            target_supercell: Target supercell structure
            normalize: Whether to normalize the projection coefficient
            atom_mapping: Optional mapping from source to target atoms

        Returns:
            float: Projection coefficient (normalized if normalize=True)

        Example:
            >>> # Project displacement between different supercell representations
            >>> coeff = modes.project_displacement_to_supercell(
            ...     source_disp, source_supercell, target_supercell
            ... )
            >>> print(f"Projection coefficient: {coeff:.6f}")
        """
        from phonproj.core.structure_analysis import (
            project_displacements_between_supercells,
        )

        # For this convenience method, we assume the target displacement is the same
        # physical displacement pattern but mapped to the target supercell atom ordering
        # Since we're projecting onto the same displacement pattern, we use the source
        # displacement as both source and target, but mapped appropriately

        return project_displacements_between_supercells(
            source_displacement=source_displacement,
            target_displacement=source_displacement,  # Same displacement pattern
            source_supercell=source_supercell,
            target_supercell=target_supercell,
            atom_mapping=atom_mapping,
            normalize=normalize,
        )


def create_supercell(
    primitive_cell: Atoms, supercell_matrix: np.ndarray, symprec: float = 1e-5
) -> Atoms:
    """
    Create a supercell from a primitive cell using ASE.

    Args:
        primitive_cell: ASE Atoms object for the primitive unit cell
        supercell_matrix: 3x3 transformation matrix
        symprec: Symmetry precision (not used in ASE implementation)

    Returns:
        ase.Atoms: Supercell structure
    """
    return make_supercell(primitive_cell, supercell_matrix)
