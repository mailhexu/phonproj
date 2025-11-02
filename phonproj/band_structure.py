"""
Phonon band structure representation and analysis.

This module provides the PhononBand class which extends PhononModes with additional
functionality for band structure calculations, k-path generation, and plotting.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Union, Tuple, List, Dict, Any
from pathlib import Path

from phonproj.modes import PhononModes
from phonproj.core.kpath import auto_kpath, validate_kpath, suggest_path


class PhononBand(PhononModes):
    """
    Represents phonon band structure with k-path information.

    This class extends PhononModes to include k-path data, band structure calculations,
    and plotting capabilities. It handles the generation of high-symmetry k-paths,
    calculation of phonon frequencies along those paths, and visualization.

    Attributes:
        Inherits all attributes from PhononModes plus:
        kpath_data (dict): Dictionary containing k-path information
        path (str): User-specified k-path string
        npoints (int): Number of k-points per path segment
        kpath_segments (list): List of k-path segments
        special_points (dict): Special high-symmetry points
        xcoords (list): X-coordinates for plotting
    """

    def __init__(
        self,
        primitive_cell,
        qpoints: np.ndarray,
        frequencies: np.ndarray,
        eigenvectors: np.ndarray,
        atomic_masses: Optional[np.ndarray] = None,
        gauge: str = "R",
        kpath_data: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize PhononBand object.

        Parameters
        ----------
        primitive_cell : ase.Atoms
            ASE Atoms object representing the primitive unit cell
        qpoints : np.ndarray
            Array of k-points in reciprocal space, shape (n_kpoints, 3)
        frequencies : np.ndarray
            Phonon frequencies in THz, shape (n_kpoints, n_modes)
        eigenvectors : np.ndarray
            Complex eigenvectors, shape (n_kpoints, n_modes, n_atoms*3)
        atomic_masses : np.ndarray, optional
            Atomic masses, if None uses masses from primitive_cell
        gauge : str, optional
            Gauge choice, either "R" (real) or "r" (reciprocal)
        kpath_data : dict, optional
            Dictionary containing k-path information
        """
        super().__init__(
            primitive_cell=primitive_cell,
            qpoints=qpoints,
            frequencies=frequencies,
            eigenvectors=eigenvectors,
            atomic_masses=atomic_masses,
            gauge=gauge,
        )

        self.kpath_data = kpath_data or {}
        self._initialize_kpath_data()

    def _initialize_kpath_data(self):
        """Initialize k-path data from stored information."""
        if not self.kpath_data:
            # Generate basic k-path data from existing q-points
            self.kpath_data = {
                "path": None,  # Unknown path
                "npoints": len(self.qpoints),
                "special_points": {},
                "xcoords": np.linspace(0, 1, len(self.qpoints)),
                "segments": [self.qpoints],
                "kpath_labels": [],
            }

    @classmethod
    def calculate_band_structure_from_phonopy(
        cls,
        data_source: Union[str, "Phonopy"],
        path: str = "GMXMG",
        npoints: int = 50,
        units: str = "cm-1",
    ) -> "PhononBand":
        """
        Calculate phonon band structure from Phonopy data.

        This method can accept either:
        - A path to a Phonopy YAML file
        - A Phonopy object directly

        Parameters
        ----------
        data_source : str or Phonopy
            Either a path to a Phonopy YAML file or a Phonopy object
        path : str, optional
            High-symmetry k-path (e.g., 'GMXMG', 'Γ-X-M-Γ-R')
        npoints : int, optional
            Number of k-points along the path
        units : str, optional
            Frequency units ('THz', 'cm-1', 'meV')

        Returns
        -------
        PhononBand
            Calculated band structure
        """
        try:
            from phonopy import Phonopy, load
        except ImportError:
            raise ImportError(
                "Phonopy is required for band structure calculations. Install with: pip install phonopy"
            )

        # Get Phonopy object from data source
        import os

        if isinstance(data_source, str):
            if os.path.isdir(data_source):
                # Use our improved loader that handles FORCE_SETS correctly
                from phonproj.core.io import load_from_phonopy_files

                data = load_from_phonopy_files(data_source)
                phonopy = data["phonopy"]
            else:
                # Load from YAML file
                phonopy = load(str(data_source))
        else:
            # Assume it's already a Phonopy object
            phonopy = data_source

        # --- Begin: k-path logic adapted from refs/kpath.py ---
        import numpy as np
        from ase.cell import Cell
        from ase.dft.kpoints import bandpath

        def group_band_path(bp, eps: float = 1e-8, shift: float = 0.15):
            xs, Xs, knames = bp.get_linear_kpoint_axis()
            kpts = bp.kpts
            m = xs[1:] - xs[:-1] < eps
            segments = [0] + list(np.where(m)[0] + 1) + [len(xs)]
            xlist, kptlist = [], []
            for i, (start, end) in enumerate(zip(segments[:-1], segments[1:])):
                kptlist.append(kpts[start:end])
                xlist.append(xs[start:end] + i * shift)
            m = Xs[1:] - Xs[:-1] < eps
            s = np.where(m)[0] + 1
            for i in s:
                Xs[i:] += shift
            return xlist, kptlist, Xs, knames

        def auto_kpath(
            cell: np.ndarray,
            path: str = None,
            npoints: int = 100,
            eps: float = 1e-5,
            supercell_matrix: np.ndarray = None,
        ):
            if not isinstance(cell, Cell):
                cell = Cell(cell)
            if path is None:
                bp = cell.bandpath(npoints=npoints)
                spk = bp.special_points
                xlist, kptlist, Xs, knames = group_band_path(bp)
            else:
                path = path.replace("Γ", "G")
                bp = cell.bandpath(path=path, npoints=npoints, eps=eps)
                spk = bp.special_points
                xlist, kptlist, Xs, knames = group_band_path(bp)
            if supercell_matrix is not None:
                kptlist = [np.dot(k, supercell_matrix) for k in kptlist]
                spk = {name: np.dot(k, supercell_matrix) for name, k in spk.items()}
            return xlist, kptlist, Xs, knames, spk

        # --- End: k-path logic ---

        # Convert Phonopy primitive cell to ASE Atoms and then to ASE Cell
        try:
            from phonproj.core.io import phonopy_to_ase

            primitive_cell = phonopy_to_ase(phonopy.primitive)
        except ImportError:
            from ase import Atoms

            primitive_cell = Atoms(
                symbols=phonopy.primitive.symbols,
                positions=phonopy.primitive.positions,
                cell=phonopy.primitive.cell,
                pbc=True,
            )
        cell = Cell(primitive_cell.get_cell())
        xlist, kptlist, Xs, knames, spk = auto_kpath(cell, path=path, npoints=npoints)

        cell = Cell(primitive_cell.get_cell())
        # Optionally, get supercell_matrix if available (not used here)
        xlist, kptlist, Xs, knames, spk = auto_kpath(cell, path=path, npoints=npoints)

        # Combine all k-points
        kpoints = np.concatenate(kptlist)

        # Calculate phonons directly at each k-point
        try:
            frequencies, eigenvectors = cls._calculate_phonons_at_kpoints(
                phonopy, kpoints
            )
        except RuntimeError as e:
            if "Force constants are not prepared" in str(e):
                phonopy.produce_force_constants()
                frequencies, eigenvectors = cls._calculate_phonons_at_kpoints(
                    phonopy, kpoints
                )
            else:
                raise

        # Convert units if needed
        if units != "THz":
            frequencies = cls._convert_frequencies(frequencies, units)

        # Create k-path labels
        kpath_labels = cls._create_kpath_labels(xlist, kptlist, knames, spk)

        # Create k-path data
        kpath_data = {
            "path": path,
            "npoints": npoints,
            "special_points": spk,
            "xcoords": xlist,
            "segments": kptlist,
            "kpath_labels": kpath_labels,
            "units": units,
        }

        # Create atomic masses
        atomic_masses = np.array(phonopy.primitive.masses)

        return cls(
            primitive_cell=primitive_cell,
            qpoints=kpoints,
            frequencies=frequencies,
            eigenvectors=eigenvectors,
            atomic_masses=atomic_masses,
            kpath_data=kpath_data,
        )

    @staticmethod
    def _calculate_phonons_at_kpoints(
        phonopy, kpoints: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate phonon frequencies and eigenvectors at specified k-points.
        """
        n_kpoints = len(kpoints)
        n_atoms = len(phonopy.primitive)
        n_modes = n_atoms * 3

        frequencies = np.zeros((n_kpoints, n_modes))
        eigenvectors = np.zeros((n_kpoints, n_modes, n_modes), dtype=complex)

        for i, q in enumerate(kpoints):
            dm = phonopy.get_dynamical_matrix_at_q(q)
            eigenvalues, eigenvectors_q = np.linalg.eigh(dm)
            frequencies[i] = (
                np.sqrt(np.abs(eigenvalues))
                * np.sign(eigenvalues)
                * phonopy.unit_conversion_factor
            )
            eigenvectors[i] = eigenvectors_q.T
        return frequencies, eigenvectors

    @classmethod
    def from_modes_with_path(
        cls,
        modes: PhononModes,
        path: Optional[str] = None,
        npoints: int = 100,
        units: str = "cm-1",
    ) -> "PhononBand":
        """
        Create PhononBand from PhononModes with specified k-path.

        **Deprecated**: Use calculate_band_structure_from_phonopy instead.
        This method is kept for backward compatibility.

        Parameters
        ----------
        modes : PhononModes
            PhononModes object containing phonon data
        path : str, optional
            User-specified k-path (e.g., 'GXMGR', 'Γ-X-M-Γ-R')
        npoints : int, optional
            Number of k-points along the path
        units : str, optional
            Frequency units ('THz', 'cm-1', 'meV')

        Returns
        -------
        PhononBand
            PhononBand object with calculated band structure

        Raises
        ------
        ValueError
            If Phonopy object is not available for direct calculations
        """
        # For backward compatibility, create a dummy Phonopy object if possible
        # This method is deprecated and may not work without a Phonopy object
        raise ValueError(
            "from_modes_with_path is deprecated. Use calculate_band_structure_from_phonopy(yaml_file) instead. "
            "The PhononModes object no longer stores a Phonopy object for better separation of concerns."
        )

    @classmethod
    def from_phonopy_directory(
        cls,
        directory: str,
        path: str = "GMXMG",
        npoints: int = 50,
        units: str = "THz",
        symprec: float = 1e-5,
        angle_tolerance: float = -1.0,
        **kwargs,
    ) -> "PhononBand":
        """
        Create PhononBand object from a directory containing Phonopy files.

        This method provides a convenient way to load phonon data from a directory
        containing individual Phonopy files (phonopy.yaml, FORCE_SETS, POSCAR) and
        automatically generate a band structure along a specified k-path.

        Args:
            directory: Path to directory containing Phonopy calculation files
            path: High-symmetry k-path (e.g., 'GMXMG', 'Γ-X-M-Γ-R')
            npoints: Number of k-points along the path
            units: Frequency units ('THz', 'cm-1', 'meV')
            symprec: Symmetry precision for k-path generation
            angle_tolerance: Angle tolerance for symmetry
            **kwargs: Additional arguments passed to load_from_phonopy_files

        Returns:
            PhononBand object with loaded band structure data

        Raises:
            ValueError: If directory not found or loading fails

        Example:
            >>> # Load band structure from directory
            >>> band = PhononBand.from_phonopy_directory("/path/to/calc", path='GMXMG')
            >>>
            >>> # With custom settings
            >>> band = PhononBand.from_phonopy_directory(
            ...     "/path/to/calc",
            ...     path='GMXGMG',
            ...     npoints=100,
            ...     units='cm-1'
            ... )
        """
        from ..phonon_utils import load_from_phonopy_files

        # Load phonopy data from directory
        data = load_from_phonopy_files(directory, **kwargs)
        phonopy = data["phonopy"]
        has_forces = data.get("has_forces", False)
        # Check if force constants are available
        force_constants_available = (
            hasattr(phonopy, "force_constants") and phonopy.force_constants is not None
        )

        if not (has_forces or force_constants_available):
            raise RuntimeError(
                f"No force constants available in {directory}. "
                "Displacement dataset is missing forces and FORCE_SETS is not present or invalid. "
                "Cannot compute force constants or band structure. "
                "Please provide a complete displacement dataset with forces, or use a directory with precomputed force constants (FORCE_SETS)."
            )

        # Generate k-path using existing kpath functionality
        from .kpath import auto_kpath

        # Convert Phonopy primitive cell to ASE Atoms
        from ..core.io import phonopy_to_ase

        primitive_cell = phonopy_to_ase(phonopy.primitive)

        # Generate k-path
        xlist, kptlist, Xs, knames, spk = auto_kpath(
            primitive_cell.get_cell(), path=path, npoints=npoints
        )

        # Combine all k-points
        kpoints = np.concatenate(kptlist)

        # Try to calculate phonons directly at each k-point
        # This will work if force constants are available
        from ..phonon_utils import _calculate_phonons_at_kpoints

        try:
            frequencies, eigenvectors = _calculate_phonons_at_kpoints(phonopy, kpoints)
        except RuntimeError as e:
            if "Force constants are not prepared" in str(e):
                # Try to produce force constants from displacement dataset
                try:
                    phonopy.produce_force_constants()
                    # Now try again to calculate phonons
                    frequencies, eigenvectors = _calculate_phonons_at_kpoints(
                        phonopy, kpoints
                    )
                except Exception as fc_error:
                    raise RuntimeError(
                        f"Could not produce force constants in {directory}. "
                        f"Error: {fc_error}. "
                        f"The from_phonopy_directory method requires either pre-calculated force constants "
                        f"or complete displacement data with forces. "
                        f"Available files: phonopy.yaml, FORCE_SETS, POSCAR."
                    )
            else:
                raise

        # Convert units if needed
        if units != "THz":
            from ..phonon_utils import _convert_frequencies

            frequencies = _convert_frequencies(frequencies, units)

        # Create k-path labels
        kpath_labels = cls._create_kpath_labels(xlist, kptlist, knames, spk)

        # Create k-path data
        kpath_data = {
            "path": path,
            "npoints": npoints,
            "special_points": spk,
            "xcoords": xlist,
            "segments": kptlist,
            "kpath_labels": kpath_labels,
            "units": units,
        }

        # Create atomic masses
        atomic_masses = np.array(phonopy.primitive.masses)

        # Create and return PhononBand object
        return cls(
            primitive_cell=primitive_cell,
            qpoints=kpoints,
            frequencies=frequencies,
            eigenvectors=eigenvectors,
            atomic_masses=atomic_masses,
            kpath_data=kpath_data,
        )

    @staticmethod
    def _create_kpath_labels(xlist, kptlist, knames, spk):
        """Create k-path labels for plotting."""
        current_pos = 0
        kpath_labels = []

        for i, (x, k) in enumerate(zip(xlist, kptlist)):
            for name in knames:
                # Find matches for this special point in the current segment
                matches = np.where((k == spk[name]).all(axis=1))[0]
                if matches.size > 0:
                    # For each match, add the global index
                    for match_idx in matches:
                        global_idx = match_idx + current_pos
                        kpath_labels.append((global_idx, name))
            current_pos += len(k)

        # Sort labels by index and remove duplicates at the same index
        kpath_labels.sort(key=lambda x: x[0])

        # Remove duplicate labels at the same index, keeping the first occurrence
        seen_indices = set()
        filtered_labels = []
        for idx, name in kpath_labels:
            if idx not in seen_indices:
                seen_indices.add(idx)
                filtered_labels.append((idx, name))

        return filtered_labels

    @staticmethod
    def _calculate_phonons_directly(
        modes: "PhononModes", kpoints: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate phonon frequencies and eigenvectors directly at specified k-points.

        This method uses Phonopy's dynamical matrix to calculate exact phonon properties
        at the required k-points without any interpolation.

        Parameters
        ----------
        modes : PhononModes
            PhononModes object containing the phonon calculation setup
        kpoints : np.ndarray
            Array of k-point coordinates, shape (n_kpoints, 3)

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Frequencies (THz) and eigenvectors for all k-points

        Raises
        ------
        ValueError
            If Phonopy object is not available for direct calculations
        """
        if modes.phonopy_object is None:
            raise ValueError(
                "Direct phonon calculations require a Phonopy object. "
                "Please ensure your PhononModes object was created from a Phonopy calculation "
                "(e.g., using read_phonopy_yaml)."
            )

        n_kpoints = len(kpoints)
        n_modes = modes.n_modes
        n_atoms = modes.n_atoms

        frequencies = np.zeros((n_kpoints, n_modes))
        eigenvectors = np.zeros((n_kpoints, n_modes, n_atoms * 3), dtype=complex)

        for i, q in enumerate(kpoints):
            try:
                # Get dynamical matrix at q-point using Phonopy
                dm = modes.phonopy_object.get_dynamical_matrix_at_q(q)

                # Solve eigenvalue problem: D * psi = omega^2 * psi
                eigenvalues, eigenvectors_q = np.linalg.eigh(dm)

                # Convert eigenvalues to frequencies (THz)
                # omega = 2π * frequency, so frequency = omega / (2π)
                # But Phonopy's eigenvalues are already omega^2, and frequencies are sqrt(omega^2) / (2π)
                # However, Phonopy's unit conversion factor handles this, so we use it directly
                frequencies[i] = (
                    np.sqrt(np.abs(eigenvalues))
                    * np.sign(eigenvalues)
                    * modes.phonopy_object.unit_conversion_factor
                )

                # eigenvectors_q has shape (n_atoms*3, n_atoms*3), we need (n_modes, n_atoms*3)
                eigenvectors[i] = eigenvectors_q.T

            except Exception as e:
                raise RuntimeError(
                    f"Failed to calculate phonons at k-point {q}: {e}\n"
                    "This may be due to an issue with the Phonopy calculation setup or an invalid k-point."
                ) from e

        return frequencies, eigenvectors

    @staticmethod
    def _convert_frequencies(frequencies: np.ndarray, target_units: str) -> np.ndarray:
        """
        Convert frequencies between different units.

        Parameters
        ----------
        frequencies : np.ndarray
            Frequencies in THz
        target_units : str
            Target units ('THz', 'cm-1', 'meV')

        Returns
        -------
        np.ndarray
            Converted frequencies
        """
        if target_units == "THz":
            return frequencies
        elif target_units == "cm-1":
            # 1 THz = 33.356 cm^-1
            return frequencies * 33.356
        elif target_units == "meV":
            # 1 THz = 4.1357 meV
            return frequencies * 4.1357
        else:
            raise ValueError(f"Unsupported units: {target_units}")

    def plot(
        self,
        ax: Optional[plt.Axes] = None,
        color: str = "blue",
        linewidth: float = 1.5,
        frequency_range: Optional[Tuple[float, float]] = None,
        figsize: Tuple[float, float] = (8, 6),
        dpi: int = 100,
        **kwargs,
    ) -> plt.Axes:
        """
        Plot the phonon band structure.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            Axes for plotting. If None, creates new figure
        color : str, optional
            Color for frequency bands. Default is 'blue'
        linewidth : float, optional
            Line width for bands. Default is 1.5
        frequency_range : tuple, optional
            (min_freq, max_freq) for y-axis limits. If None, auto-scale
        figsize : tuple, optional
            Figure size (width, height). Default is (8, 6)
        dpi : int, optional
            Figure DPI. Default is 100
        **kwargs
            Additional keyword arguments passed to plotting functions

        Returns
        -------
        matplotlib.axes.Axes
            The plotting axes
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

        # Filter kwargs to only include matplotlib-compatible parameters
        # Remove any non-matplotlib parameters that might have been passed
        valid_kwargs = {
            k: v
            for k, v in kwargs.items()
            if k not in ["show_imaginary", "imaginary_color"]
        }

        # Handle segmented x-coordinates
        if isinstance(self.kpath_data["xcoords"], list):
            # Segmented plotting
            for i, segment_x in enumerate(self.kpath_data["xcoords"]):
                # Get corresponding frequency segment
                start_idx = sum(len(s) for s in self.kpath_data["xcoords"][:i])
                end_idx = start_idx + len(segment_x)
                segment_frequencies = self.frequencies[start_idx:end_idx]

                # Plot each band in this segment
                for band_idx in range(segment_frequencies.shape[1]):
                    ax.plot(
                        segment_x,
                        segment_frequencies[:, band_idx],
                        color=color,
                        linewidth=linewidth,
                        **valid_kwargs,
                    )
        else:
            # Continuous plotting - plot each band separately
            for band_idx in range(self.frequencies.shape[1]):
                ax.plot(
                    self.kpath_data["xcoords"],
                    self.frequencies[:, band_idx],
                    color=color,
                    linewidth=linewidth,
                    **valid_kwargs,
                )

        # Set special point labels
        if self.kpath_data["kpath_labels"]:
            label_indices, label_names = zip(*self.kpath_data["kpath_labels"])
            ax.set_xticks([self._get_xcoord_at_index(i) for i in label_indices])

            # Convert 'G' to 'Γ' for display
            display_names = ["Γ" if name == "G" else name for name in label_names]
            ax.set_xticklabels(display_names)

        # Add y=0 line for reference
        ax.axhline(y=0, color="black", linestyle="--", linewidth=0.8, alpha=0.7)

        # Set axis labels and title
        units = self.kpath_data.get("units", "THz")
        ax.set_xlabel("k-path", fontsize=12)
        ax.set_ylabel(f"Frequency ({units})", fontsize=12)
        ax.set_title("Phonon Band Structure", fontsize=14, fontweight="bold")

        # Set frequency range if specified
        if frequency_range is not None:
            ax.set_ylim(frequency_range)
        else:
            # Smart auto-scaling based on units and frequency content
            units = self.kpath_data.get("units", "THz")
            min_freq = np.min(self.frequencies)
            max_freq = np.max(self.frequencies)

            # Always include the full frequency range (including imaginary) by default
            # Use percentage-based margins for optimal scaling
            freq_range = abs(max_freq - min_freq)
            margin = 0.05 * freq_range  # 5% margin on each side

            # Ensure minimum margin for very small ranges
            if units == "cm-1":
                margin = max(margin, 5.0)  # Minimum 5 cm-1 margin
            elif units == "meV":
                margin = max(margin, 2.0)  # Minimum 2 meV margin
            else:  # THz
                margin = max(margin, 0.01)  # Minimum 0.01 THz margin

            ax.set_ylim(min_freq - margin, max_freq + margin)

        # Add grid
        ax.grid(True, alpha=0.3, linestyle="-", linewidth=0.5)

        # Improve layout
        ax.tick_params(axis="both", which="major", labelsize=10)
        plt.tight_layout()

        return ax

    def _get_xcoord_at_index(self, index: int) -> float:
        """Get the x-coordinate for a given k-point index."""
        xcoords = self.kpath_data["xcoords"]
        if isinstance(xcoords, list):
            # Segmented x-coordinates
            current_idx = 0
            for segment in xcoords:
                if current_idx + len(segment) > index:
                    return segment[index - current_idx]
                current_idx += len(segment)
            raise IndexError(f"Index {index} out of range")
        else:
            # Continuous x-coordinates
            return xcoords[index]

    def save_data(self, filename: str, format: str = "json"):
        """
        Save band structure data to file.

        Parameters
        ----------
        filename : str
            Output filename
        format : str, optional
            File format ('json' or 'csv'). Default is 'json'
        """
        if format.lower() == "json":
            self._save_json(filename)
        elif format.lower() == "csv":
            self._save_csv(filename)
        else:
            raise ValueError(f"Unsupported format: {format}")

    def _save_json(self, filename: str):
        """Save data to JSON format."""
        import json

        xcoords = self.kpath_data["xcoords"]
        if isinstance(xcoords, list):
            xcoords_flat = np.concatenate(xcoords)
        else:
            xcoords_flat = xcoords

        data = {
            "energies": self.frequencies.tolist(),
            "kpoints": self.qpoints.tolist(),
            "kpath_labels": [
                [int(idx), str(label)] for idx, label in self.kpath_data["kpath_labels"]
            ],
            "special_points": {
                str(k): v.tolist() for k, v in self.kpath_data["special_points"].items()
            },
            "xcoords": [
                x.tolist() if isinstance(x, np.ndarray) else x
                for x in (xcoords if isinstance(xcoords, list) else [xcoords])
            ],
            "lattice": self.primitive_cell.get_cell().tolist(),
            "units": self.kpath_data.get("units", "THz"),
            "path": self.kpath_data.get("path"),
            "metadata": {
                "n_kpoints": int(len(self.qpoints)),
                "n_modes": int(self.frequencies.shape[1]),
                "frequency_range": [
                    float(np.min(self.frequencies)),
                    float(np.max(self.frequencies)),
                ],
            },
        }

        with open(filename, "w") as f:
            json.dump(data, f, indent=2)

    def _save_csv(self, filename: str):
        """Save frequency data to CSV format."""
        xcoords = self.kpath_data["xcoords"]
        if isinstance(xcoords, list):
            x_flat = np.concatenate(xcoords)
        else:
            x_flat = xcoords

        # Create header
        header = ["k_x", "k_y", "k_z", "x_coord"] + [
            f"band_{i + 1}" for i in range(self.frequencies.shape[1])
        ]

        # Combine data
        data = np.column_stack([self.qpoints, x_flat, self.frequencies])

        # Save to CSV
        np.savetxt(filename, data, delimiter=",", header=",".join(header), comments="")

    @property
    def path(self) -> Optional[str]:
        """Get the k-path string."""
        return self.kpath_data.get("path")

    @property
    def special_points(self) -> Dict[str, np.ndarray]:
        """Get special high-symmetry points."""
        return self.kpath_data.get("special_points", {})

    @property
    def xcoords(self) -> Union[np.ndarray, List[np.ndarray]]:
        """Get x-coordinates for plotting."""
        return self.kpath_data.get("xcoords", [])

    @property
    def units(self) -> str:
        """Get frequency units."""
        return self.kpath_data.get("units", "THz")
