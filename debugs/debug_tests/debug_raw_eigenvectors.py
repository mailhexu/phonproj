#!/usr/bin/env python3
"""
Debug raw phonopy eigenvector orthogonality.
"""

import numpy as np
from pathlib import Path
from phonproj.modes import PhononModes

# Load BaTiO3 data
BATIO3_YAML_PATH = Path("data/BaTiO3_phonopy_params.yaml")


def debug_raw_eigenvectors():
    """Debug raw phonopy eigenvector orthogonality."""

    # Load data with a single Gamma q-point to test fundamental orthogonality
    qpoints = np.array([[0.0, 0.0, 0.0]])
    modes = PhononModes.from_phonopy_yaml(str(BATIO3_YAML_PATH), qpoints)

    print("=== Raw Phonopy Eigenvector Orthogonality ===")

    # Get all eigenvectors for Gamma point
    n_modes = len(modes.frequencies[0])
    eigenvectors = []

    for mode_idx in range(n_modes):
        freq, eigvec = modes.get_mode(0, mode_idx)
        eigenvectors.append(eigvec)
        print(f"Mode {mode_idx}: frequency = {freq:.6f}")

    eigenvectors = np.array(eigenvectors)
    print(f"Eigenvector shape: {eigenvectors.shape}")

    # Check orthogonality of raw eigenvectors (no mass weighting)
    print("\n=== Raw Eigenvector Orthogonality (no mass weighting) ===")
    max_projection = 0.0
    for i in range(min(10, n_modes)):
        for j in range(i + 1, min(10, n_modes)):
            # Simple dot product (no mass weighting)
            projection = np.real(np.vdot(eigenvectors[i], eigenvectors[j]))
            max_projection = max(max_projection, abs(projection))
            if abs(projection) > 1e-12:
                print(f"Raw modes {i},{j}: projection = {projection}")
    print(f"Raw max projection: {max_projection}")

    # Check orthogonality with mass weighting (our approach)
    print("\n=== Mass-Weighted Eigenvector Orthogonality ===")
    max_projection_mw = 0.0
    for i in range(min(10, n_modes)):
        for j in range(i + 1, min(10, n_modes)):
            # Apply mass weighting like in get_eigen_displacement
            eigvec_i = eigenvectors[i].reshape(modes._n_atoms, 3)
            eigvec_j = eigenvectors[j].reshape(modes._n_atoms, 3)

            # Apply gauge
            if modes.gauge == "R":
                eigvec_i = eigvec_i.real
                eigvec_j = eigvec_j.real
            else:
                eigvec_i = eigvec_i.imag
                eigvec_j = eigvec_j.imag

            # Apply mass weighting: u = e / sqrt(m)
            mass_weights = np.sqrt(modes.atomic_masses)
            eigvec_i = eigvec_i / mass_weights[:, np.newaxis]
            eigvec_j = eigvec_j / mass_weights[:, np.newaxis]

            # Mass-weighted projection
            projection = modes.mass_weighted_projection(eigvec_i, eigvec_j)
            max_projection_mw = max(max_projection_mw, abs(projection))
            if abs(projection) > 1e-12:
                print(f"Mass-weighted modes {i},{j}: projection = {projection}")

    print(f"Mass-weighted max projection: {max_projection_mw}")

    # Test what get_eigen_displacement produces
    print("\n=== get_eigen_displacement Orthogonality ===")
    displacements = []
    for mode_idx in range(min(10, n_modes)):
        disp = modes.get_eigen_displacement(0, mode_idx, normalize=True)
        displacements.append(disp)

    max_projection_ged = 0.0
    for i in range(len(displacements)):
        for j in range(i + 1, len(displacements)):
            projection = modes.mass_weighted_projection(
                displacements[i], displacements[j]
            )
            max_projection_ged = max(max_projection_ged, abs(projection))
            if abs(projection) > 1e-12:
                print(
                    f"get_eigen_displacement modes {i},{j}: projection = {projection}"
                )

    print(f"get_eigen_displacement max projection: {max_projection_ged}")


if __name__ == "__main__":
    debug_raw_eigenvectors()
