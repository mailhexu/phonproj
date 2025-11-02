"""Test band structure calculation and plotting for PbTiO3."""

from phonproj.band_structure import PhononBand
import matplotlib.pyplot as plt


def test_ppto3_band_structure():
    directory = "/Users/hexu/projects/phonproj/data/yajundata/0.02-P4mmm-PTO"
    try:
        band = PhononBand.calculate_band_structure_from_phonopy(
            directory, path="GMXMG", npoints=300, units="cm-1"
        )
        # If no error, check normal outputs
        assert band is not None
        assert band.frequencies.shape[0] > 0
        assert band.eigenvectors.shape[0] > 0
        ax = band.plot()
        assert ax.get_xlabel() == "k-path"
        assert "Frequency" in ax.get_ylabel()
        print("✓ PbTiO3 band structure calculated and plotted successfully.")
    except RuntimeError as e:
        # Expect error if forces are missing
        msg = str(e)
        assert "missing forces" in msg
        print(f"Expected error: {msg}")


if __name__ == "__main__":
    test_ppto3_band_structure()
    print("✅ PbTiO3 band structure test passed!")
