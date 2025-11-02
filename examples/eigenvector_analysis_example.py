"""
Example: Eigenvector Analysis and Orthonormality Testing

This example demonstrates how to:
1. Extract eigenvectors at specific q-points
2. Verify that eigenvectors form an orthonormal basis
3. Understand the mathematical properties of phonon eigenvectors

The orthonormality condition for eigenvectors is:
    <e_i|e_j> = e_i† @ e_j = δ_ij

where δ_ij is the Kronecker delta (1 if i=j, 0 otherwise).
"""

import numpy as np
from phonproj.band_structure import PhononBand


def check_eigenvector_orthonormality(band, q_index=0, tolerance=1e-10):
    """
    Check if eigenvectors at a given q-point are orthonormal.

    This function demonstrates how to use the package method
    check_eigenvector_orthonormality() for verification.

    Parameters
    ----------
    band : PhononBand
        PhononBand object with calculated eigenvectors
    q_index : int
        Index of the q-point to check
    tolerance : float
        Numerical tolerance for orthonormality check

    Returns
    -------
    bool
        True if eigenvectors are orthonormal

    Raises
    ------
    AssertionError
        If eigenvectors are not orthonormal
    """
    print(f"\n{'='*60}")
    print(f"Eigenvector Orthonormality Check at q-point: {q_index}")
    print(f"{'='*60}")

    print(f"Using package method: band.check_eigenvector_orthonormality()")
    print(f"Tolerance: {tolerance:.0e}")

    # Use the package method - this is the clean, reusable way!
    is_orthonormal, max_error, errors = band.check_eigenvector_orthonormality(
        q_index, tolerance=tolerance, verbose=True
    )

    # Check the results
    if is_orthonormal:
        print(f"\n✓ Eigenvectors ARE orthonormal!")
        return True
    else:
        print(f"\n✗ Eigenvectors are NOT orthonormal!")
        raise AssertionError(
            f"Eigenvectors not orthonormal: max_error={max_error:.2e} > tolerance={tolerance:.0e}"
        )


def analyze_eigenvector_properties(band, q_index=0, mode_index=0):
    """
    Analyze properties of a specific eigenvector.

    Parameters
    ----------
    band : PhononBand
        PhononBand object
    q_index : int
        Index of q-point
    mode_index : int
        Index of phonon mode
    """
    eigenvector = band.eigenvectors[q_index, mode_index]
    frequency = band.frequencies[q_index, mode_index]

    print(f"\n{'='*60}")
    print(f"Eigenvector Analysis - Mode {mode_index} at q-index {q_index}")
    print(f"{'='*60}")

    # Basic properties
    print(f"\nEigenvector properties:")
    print(f"  Frequency: {frequency:.2f} cm⁻¹")
    print(f"  Shape: {eigenvector.shape}")
    print(f"  Data type: {eigenvector.dtype}")

    # Normalization check
    norm = np.vdot(eigenvector, eigenvector)
    print(f"\nNormalization:")
    print(f"  <e|e> = {norm.real:.10f} + {norm.imag:.10f}i")
    print(f"  Should be: 1.0 + 0.0i")

    # Phase information
    phases = np.angle(eigenvector)
    print(f"\nPhase information:")
    print(f"  Phase range: [{phases.min():.3f}, {phases.max():.3f}] radians")
    print(f"  Unique phases: {len(np.unique(np.round(phases, 2)))}")

    # Real/imaginary parts
    print(f"\nReal/Imaginary components:")
    print(f"  Max real part: {eigenvector.real.max():.6f}")
    print(f"  Min real part: {eigenvector.real.min():.6f}")
    print(f"  Max imag part: {eigenvector.imag.max():.6f}")
    print(f"  Min imag part: {eigenvector.imag.min():.6f}")


def main():
    """
    Main example demonstrating eigenvector analysis.
    """
    print("\n" + "="*70)
    print("Phonon Eigenvector Analysis Example")
    print("="*70)

    # Load BaTiO3 phonon data
    yaml_path = "/Users/hexu/projects/phonproj/data/BaTiO3_phonopy_params.yaml"

    print(f"\nLoading phonon data from: {yaml_path}")
    band = PhononBand.calculate_band_structure_from_phonopy(
        yaml_path,
        path="GMXMG",
        npoints=30,
        units="cm-1"
    )

    print(f"✓ Data loaded successfully!")
    print(f"  - Number of q-points: {band.frequencies.shape[0]}")
    print(f"  - Number of modes: {band.frequencies.shape[1]}")
    print(f"  - Number of atoms: {len(band.primitive_cell)}")

    # Check orthonormality at Γ-point (q-index 0)
    check_eigenvector_orthonormality(band, q_index=0)

    # Check orthonormality at M-point (find from labels)
    m_indices = [idx for idx, label in band.kpath_data['kpath_labels'] if label == 'M']
    if m_indices:
        print(f"\nFound M-point at index: {m_indices[0]}")
        check_eigenvector_orthonormality(band, q_index=m_indices[0])

    # Analyze specific eigenvector
    analyze_eigenvector_properties(band, q_index=0, mode_index=0)

    # Show frequency range
    print(f"\n{'='*60}")
    print("Phonon Mode Summary")
    print(f"{'='*60}")
    print(f"\nFrequency statistics at Γ-point:")
    gamma_frequencies = band.frequencies[0]
    print(f"  Minimum: {gamma_frequencies.min():.2f} cm⁻¹")
    print(f"  Maximum: {gamma_frequencies.max():.2f} cm⁻¹")
    print(f"  Mean: {gamma_frequencies.mean():.2f} cm⁻¹")

    # Count acoustic vs optical modes
    acoustic_threshold = 50.0  # cm⁻¹
    n_acoustic = np.sum(gamma_frequencies < acoustic_threshold)
    n_optical = np.sum(gamma_frequencies >= acoustic_threshold)

    print(f"\nMode classification (at Γ-point):")
    print(f"  Acoustic modes (<{acoustic_threshold} cm⁻¹): {n_acoustic}")
    print(f"  Optical modes (≥{acoustic_threshold} cm⁻¹): {n_optical}")

    print("\n" + "="*70)
    print("Example completed successfully!")
    print("="*70)

    # Don't show plot for this example (band structure example already does)
    print("\nNote: Run 'phonon_band_structure_simple_example.py' to see plotting!")


if __name__ == "__main__":
    main()