"""
Example: Eigendisplacement Analysis and Mass-Weighted Properties

This example demonstrates:
1. How to extract eigendisplacement patterns from phonon modes
2. Computing mass-weighted norms of eigendisplacements
3. Understanding the relationship between eigenvectors and eigendisplacements
4. Physical interpretation of mass-weighted eigendisplacements

The mass-weighted norm condition is:
    <u|M|u> = Σ_i m_i * |u_i|² = 1

where u_i is the displacement of atom i and m_i is its mass.
"""

import numpy as np
from phonproj.band_structure import PhononBand


def analyze_eigendisplacement(band, q_index=0, mode_index=0):
    """
    Analyze an eigendisplacement pattern.

    Parameters
    ----------
    band : PhononBand
        PhononBand object with calculated eigenvectors
    q_index : int
        Index of the q-point
    mode_index : int
        Index of the phonon mode
    """
    modes = band

    # Get eigendisplacement
    eigendisp = modes.get_eigen_displacement(q_index, mode_index)

    # Get frequency for reference
    frequency = modes.frequencies[q_index, mode_index]

    print(f"\n{'='*60}")
    print(f"Eigendisplacement Analysis - Mode {mode_index} at q-index {q_index}")
    print(f"{'='*60}")

    print(f"\nPhonon mode properties:")
    print(f"  Frequency: {frequency:.2f} cm⁻¹")
    print(f"  Number of atoms: {len(band.primitive_cell)}")

    # Determine mode type
    if abs(frequency) < 1.0:
        mode_type = "Acoustic (or zero-frequency optical)"
    elif frequency < 0:
        mode_type = "Imaginary (unstable)"
    else:
        mode_type = "Optical"

    print(f"  Mode type: {mode_type}")

    # Calculate mass-weighted norm
    mass_weighted_norm = modes.mass_weighted_norm(eigendisp)

    print(f"\nMass-weighted norm:")
    print(f"  <u|M|u> = {mass_weighted_norm:.10f}")
    print(f"  Expected: 1.0")
    print(f"  Error: {abs(mass_weighted_norm - 1.0):.2e}")

    # Show displacement pattern
    print(f"\nEigendisplacement pattern:")
    print(f"  Shape: {eigendisp.shape} (n_atoms × 3)")
    print(f"  Atomic displacements (Å):")

    atomic_masses = band.atomic_masses
    for atom_idx in range(len(band.primitive_cell)):
        symbol = band.primitive_cell[atom_idx].symbol
        mass = atomic_masses[atom_idx]
        dx, dy, dz = eigendisp[atom_idx]
        displacement_magnitude = np.sqrt(dx**2 + dy**2 + dz**2)

        print(f"    Atom {atom_idx:2d} ({symbol:2s}, mass={mass:7.3f}): "
              f"[{dx:8.5f}, {dy:8.5f}, {dz:8.5f}] |u|={displacement_magnitude:.5f}")

    # Show which atoms move most
    max_disp_idx = np.argmax(np.linalg.norm(eigendisp, axis=1))
    max_symbol = band.primitive_cell[max_disp_idx].symbol
    max_disp = np.linalg.norm(eigendisp[max_disp_idx])

    print(f"\nLargest displacement:")
    print(f"  Atom {max_disp_idx} ({max_symbol}): {max_disp:.5f} Å")

    return eigendisp


def compare_eigenvector_vs_eigendisplacement(band, q_index=0, mode_index=0):
    """
    Compare eigenvector and eigendisplacement for understanding.
    """
    print(f"\n{'='*60}")
    print(f"Eigenvector vs Eigendisplacement Comparison")
    print(f"{'='*60}")

    # Get raw eigenvector
    eigenvector = band.eigenvectors[q_index, mode_index]

    # Get eigendisplacement
    eigendisp = band.get_eigen_displacement(q_index, mode_index)

    # Flatten for comparison
    eigendisp_flat = eigendisp.flatten()

    print(f"\nMode {mode_index}:")
    print(f"  Eigenvector (first 5 components): {eigenvector[:5]}")
    print(f"  Eigendisplacement flat (first 5): {eigendisp_flat[:5]}")

    # Check if eigendisp = sqrt(mass) * eigenvector
    masses_repeated = np.repeat(band.atomic_masses, 3)
    mass_scaling = np.sqrt(masses_repeated)

    predicted_disp = eigenvector * mass_scaling

    print(f"\nExpected (eigenvector × sqrt(mass)):")
    print(f"  First 5: {predicted_disp[:5]}")

    print(f"\nDifference:")
    print(f"  First 5: {eigendisp_flat[:5] - predicted_disp[:5]}")
    print(f"  Max error: {np.max(np.abs(eigendisp_flat - predicted_disp)):.2e}")

    if np.max(np.abs(eigendisp_flat - predicted_disp)) < 1e-10:
        print(f"  ✓ Eigendisplacement = sqrt(mass) × eigenvector")
    else:
        print(f"  ✗ Relationship not as expected")


def main():
    """
    Main example demonstrating eigendisplacement analysis.
    """
    print("\n" + "="*70)
    print("Eigendisplacement Analysis Example")
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

    # Analyze different types of modes
    print(f"\n{'='*70}")
    print("Analyzing Different Mode Types")
    print(f"{'='*70}")

    # Acoustic mode (lowest frequency)
    analyze_eigendisplacement(band, q_index=0, mode_index=0)

    # Optical mode (higher frequency)
    analyze_eigendisplacement(band, q_index=0, mode_index=10)

    # Show comparison between eigenvector and eigendisplacement
    compare_eigenvector_vs_eigendisplacement(band, q_index=0, mode_index=5)

    # Show mass-weighted norm for multiple modes
    print(f"\n{'='*60}")
    print("Mass-Weighted Norm for Multiple Modes")
    print(f"{'='*60}")

    modes_to_check = [0, 1, 2, 5, 10, 14]
    print(f"\nChecking modes: {modes_to_check}")

    all_normalized = True
    for mode_idx in modes_to_check:
        eigendisp = band.get_eigen_displacement(0, mode_idx)
        norm = band.mass_weighted_norm(eigendisp)
        freq = band.frequencies[0, mode_idx]
        error = abs(norm - 1.0)

        status = "✓" if error < 1e-10 else "✗"
        print(f"  Mode {mode_idx:2d} (freq={freq:7.2f} cm⁻¹): "
              f"norm={norm:.10f}, error={error:.2e} {status}")

        if error >= 1e-10:
            all_normalized = False

    if all_normalized:
        print(f"\n✓ All tested modes have unit mass-weighted norm!")
    else:
        print(f"\n✗ Some modes are not properly normalized!")

    # Summary
    print(f"\n{'='*70}")
    print("Summary")
    print(f"{'='*70}")
    print(f"\nEigendisplacements in PhononTools:")
    print(f"  • Represent mass-weighted displacement patterns")
    print(f"  • Satisfy: <u|M|u> = Σ m_i |u_i|² = 1")
    print(f"  • Derived from eigenvectors: u = √M · e")
    print(f"  • Used for supercell displacement generation")
    print(f"  • Preserve physical correctness of mass-weighted dynamics")

    print("\n" + "="*70)
    print("Example completed successfully!")
    print("="*70)


if __name__ == "__main__":
    main()