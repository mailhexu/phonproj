#!/usr/bin/env python3
"""
Debug script to analyze what phonopy API is actually doing
vs our direct method.
"""

import numpy as np
from pathlib import Path
from phonproj.modes import PhononModes
from ase.build import make_supercell

# Data path
BATIO3_YAML_PATH = Path(__file__).parent / "data" / "BaTiO3_phonopy_params.yaml"


def debug_phonopy_internals():
    """Debug what phonopy is actually doing."""

    # Set up the same test case as the failing test
    qpoints_2x2x2 = []
    for i in range(2):
        for j in range(2):
            for k in range(2):
                qpoints_2x2x2.append([i / 2.0, j / 2.0, k / 2.0])
    qpoints_2x2x2 = np.array(qpoints_2x2x2)

    modes = PhononModes.from_phonopy_yaml(str(BATIO3_YAML_PATH), qpoints_2x2x2)

    # Use 2x2x2 supercell
    supercell_matrix = np.eye(3, dtype=int) * 2

    # Find a non-Gamma q-point (same as the test)
    non_gamma_index = None
    for i, qpoint in enumerate(modes.qpoints):
        if not np.allclose(qpoint, [0, 0, 0], atol=1e-6):
            non_gamma_index = i
            break

    print(f"Using q-point index {non_gamma_index}: {modes.qpoints[non_gamma_index]}")

    # Let's investigate what phonopy is doing internally
    try:
        from phonopy import Phonopy
        from phonopy.interface.phonopy_yaml import PhonopyYaml

        # Recreate the phonopy creation process from the method
        qpoint = modes.qpoints[non_gamma_index].copy()

        ph_yaml = PhonopyYaml()
        ph_yaml.read(modes._phonopy_yaml_path)

        # Create phonopy object with our desired supercell
        ph = Phonopy(
            unitcell=ph_yaml.unitcell,
            supercell_matrix=supercell_matrix,
            primitive_matrix=ph_yaml.primitive_matrix,
        )

        # Set the force constants from the original calculation
        ph.force_constants = ph_yaml.force_constants

        print(f"\nOriginal qpoints in modes object:")
        print(f"Q-point: {qpoint}")
        print(f"Frequencies: {modes.frequencies[non_gamma_index][:2]}")
        print(f"Eigenvector shapes: {modes.eigenvectors[non_gamma_index][:2].shape}")

        # Now check what phonopy computes for this q-point
        ph.run_qpoints([qpoint])
        ph_freqs = ph.qpoints.frequencies[0]  # First q-point
        ph_eigvecs = ph.qpoints.eigenvectors[0]  # First q-point

        print(f"\nPhonopy recomputed for same q-point:")
        print(f"Q-point: {qpoint}")
        print(f"Frequencies: {ph_freqs[:2]}")
        print(f"Eigenvector shapes: {ph_eigvecs[:2].shape}")

        print(f"\nFrequency comparison:")
        print(f"Original: {modes.frequencies[non_gamma_index][:2]}")
        print(f"Phonopy:  {ph_freqs[:2]}")
        print(f"Difference: {modes.frequencies[non_gamma_index][:2] - ph_freqs[:2]}")

        # Compare eigenvectors
        orig_ev_0 = modes.eigenvectors[non_gamma_index][0]
        orig_ev_1 = modes.eigenvectors[non_gamma_index][1]
        ph_ev_0 = ph_eigvecs[0]
        ph_ev_1 = ph_eigvecs[1]

        print(f"\nEigenvector comparison (first few components):")
        print(f"Mode 0 - Original: {orig_ev_0[:3]}")
        print(f"Mode 0 - Phonopy:  {ph_ev_0[:3]}")
        print(f"Mode 1 - Original: {orig_ev_1[:3]}")
        print(f"Mode 1 - Phonopy:  {ph_ev_1[:3]}")

        # Check orthogonality of original eigenvectors
        # Note: phonopy eigenvectors are mass-weighted already
        masses = modes.atomic_masses
        mass_weights = np.repeat(masses, 3)

        orig_overlap = np.dot(np.conj(orig_ev_0), orig_ev_1 * mass_weights)
        ph_overlap = np.dot(np.conj(ph_ev_0), ph_ev_1)  # phonopy already mass-weighted

        print(f"\nEigenvector orthogonality check:")
        print(f"Original modes overlap: {orig_overlap}")
        print(f"Phonopy modes overlap:  {ph_overlap}")

        # Now test modulations for both modes
        print(f"\n=== TESTING PHONOPY MODULATIONS ===")

        for mode_idx in [0, 1]:
            phonon_modes = [[qpoint.tolist(), mode_idx, 1.0, 0.0]]
            ph.run_modulations(dimension=supercell_matrix, phonon_modes=phonon_modes)
            modulations, supercell_atoms = ph.get_modulations_and_supercell()

            if modulations.ndim == 3 and modulations.shape[0] == 1:
                displacements = modulations[0]
            else:
                displacements = modulations

            print(f"Mode {mode_idx}: displacement shape = {displacements.shape}")
            print(
                f"Mode {mode_idx}: max displacement = {np.max(np.abs(displacements))}"
            )

            # Store for orthogonality check
            if mode_idx == 0:
                mod_0 = displacements.copy()
            else:
                mod_1 = displacements.copy()

        # Check orthogonality of modulations
        # We need to apply mass-weighting
        supercell_masses = np.repeat(supercell_atoms.get_masses(), 3).reshape(-1, 3)

        def mass_weighted_projection(disp1, disp2):
            flat1 = disp1.flatten()
            flat2 = disp2.flatten()
            mass_flat = supercell_masses.flatten()
            return np.sum(np.conj(flat1) * flat2 * mass_flat)

        modulation_overlap = mass_weighted_projection(mod_0, mod_1)
        print(f"\nModulation orthogonality: {modulation_overlap}")

        # This tells us if the problem is in the modulation step or eigenvector step

    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    debug_phonopy_internals()
