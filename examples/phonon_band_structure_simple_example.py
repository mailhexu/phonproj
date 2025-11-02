"""
Minimal example demonstrating phonon band structure calculation and plotting.

This example shows how to:
1. Load phonon data from phonopy files
2. Calculate the phonon band structure
3. Plot the band structure with proper labels and formatting

The example works with BaTiO3 phonon data.
"""

from phonproj.band_structure import PhononBand


def main():
    """
    Calculate and plot phonon band structure for BaTiO3.

    This demonstrates the core workflow for band structure analysis.
    """
    # Path to BaTiO3 phonon data
    yaml_path = "/Users/hexu/projects/phonproj/data/BaTiO3_phonopy_params.yaml"

    print("=" * 60)
    print("Phonon Band Structure Calculation Example")
    print("=" * 60)
    print(f"\nLoading data from: {yaml_path}\n")

    # Calculate band structure along high-symmetry path
    # Parameters:
    # - path: High-symmetry k-path (GMXMG for perovskites)
    # - npoints: Number of k-points (30 for quick calculation)
    # - units: Frequency units (cm-1 for spectroscopic comparison)
    band = PhononBand.calculate_band_structure_from_phonopy(
        yaml_path,
        path="GMXMG",
        npoints=30,
        units="cm-1"
    )

    print(f"✓ Band structure calculated successfully!")
    print(f"  - Number of k-points: {band.frequencies.shape[0]}")
    print(f"  - Number of phonon modes: {band.frequencies.shape[1]}")
    print(f"  - Frequency range: {band.frequencies.min():.2f} to {band.frequencies.max():.2f} cm⁻¹")

    # Plot the band structure
    print("\nGenerating band structure plot...")
    ax = band.plot(
        color='blue',
        linewidth=1.5,
        figsize=(10, 6),
        dpi=100
    )

    print("✓ Band structure plot created!")
    print(f"  - x-axis: k-path")
    print(f"  - y-axis: Frequency (cm⁻¹)")
    print(f"  - Special points labeled: {', '.join([name for _, name in band.kpath_data['kpath_labels']])}")

    # Optionally save the data
    output_file = "band_structure_batio3.json"
    band.save_data(output_file, format="json")
    print(f"\n✓ Band structure data saved to: {output_file}")

    print("\n" + "=" * 60)
    print("Example completed successfully!")
    print("=" * 60)

    # Display the plot
    import matplotlib.pyplot as plt
    plt.show()


if __name__ == "__main__":
    main()