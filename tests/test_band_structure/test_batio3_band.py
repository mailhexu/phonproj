"""Test band structure calculation and plotting for BaTiO3."""

from phonproj.band_structure import PhononBand
import matplotlib.pyplot as plt


def test_batio3_band_structure():
    yaml_path = "/Users/hexu/projects/phonproj/data/BaTiO3_phonopy_params.yaml"
    band = PhononBand.calculate_band_structure_from_phonopy(
        yaml_path, path="GMXMG", npoints=30, units="cm-1"
    )
    assert band is not None
    assert band.frequencies.shape[0] > 0
    assert band.eigenvectors.shape[0] > 0
    # Plot and check axes
    ax = band.plot()
    assert ax.get_xlabel() == "k-path"
    assert "Frequency" in ax.get_ylabel()
    print("✓ BaTiO3 band structure calculated and plotted successfully.")



if __name__ == "__main__":
    test_batio3_band_structure()
    print("✅ BaTiO3 band structure test passed!")
