#!/usr/bin/env python3
"""
ISODISTORT Analysis Example

This example demonstrates how to use phonproj with ISODISTORT files to analyze
specific structural distortions and identify contributing phonon modes.

PURPOSE:
    Show how to decompose ISODISTORT distortions into phonon mode contributions
    and interpret the results for understanding structural phase transitions.

USAGE:
    uv run python examples/isodistort_example.py

EXPECTED OUTPUT:
    - Mode decomposition table showing dominant phonon contributions
    - Completeness analysis indicating reconstruction quality
    - Q-point summary showing which regions of Brillouin zone are important

FILES USED:
    - data/BaTiO3_phonopy_params.yaml - Phonopy calculation data
    - data/yajundata/P4mmm-ref.txt - ISODISTORT file with P4mmm distortion

REQUIREMENTS:
    - BaTiO3 phonopy data in data/ directory
    - ISODISTORT file from Yajun's data set
"""

import subprocess
import sys
from pathlib import Path

# Data paths
BATIO3_YAML_PATH = Path("data/BaTiO3_phonopy_params.yaml")
ISODISTORT_PATH = Path("data/yajundata/P4mmm-ref.txt")


def run_isodistort_analysis():
    """Run ISODISTORT distortion analysis using CLI."""

    print("=" * 80)
    print("ISODISTORT DISTORTION ANALYSIS EXAMPLE")
    print("=" * 80)

    # Check if data files exist
    if not BATIO3_YAML_PATH.exists():
        print(f"❌ Error: Phonopy data not found at {BATIO3_YAML_PATH}")
        print("Please ensure BaTiO3 phonopy data is available.")
        return False

    if not ISODISTORT_PATH.exists():
        print(f"❌ Error: ISODISTORT file not found at {ISODISTORT_PATH}")
        print("Please ensure ISODISTORT data is available.")
        return False

    print(f"📁 Phonopy data: {BATIO3_YAML_PATH}")
    print(f"📁 ISODISTORT file: {ISODISTORT_PATH}")
    print()

    # Build CLI command
    cmd = [
        "phonproj-decompose",
        "--phonopy",
        str(BATIO3_YAML_PATH),
        "--isodistort",
        str(ISODISTORT_PATH),
        "--supercell",
        "4x4x2",
        "--remove-com",
        "--quiet",
    ]

    print("🚀 Running command:")
    print(" ".join(cmd))
    print()

    try:
        # Run the analysis
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)

        print("✅ Analysis completed successfully!")
        print()
        print("RESULTS:")
        print("-" * 40)
        print(result.stdout)

        return True

    except subprocess.CalledProcessError as e:
        print(f"❌ Error running analysis:")
        print(f"Return code: {e.returncode}")
        print(f"Error: {e.stderr}")
        return False
    except FileNotFoundError:
        print("❌ Error: 'phonproj-decompose' command not found.")
        print("Please install phonproj with: uv pip install -e .")
        return False


def interpret_results():
    """Provide guidance on interpreting ISODISTORT results."""

    print("\n" + "=" * 80)
    print("INTERPRETING ISODISTORT RESULTS")
    print("=" * 80)

    print("""
KEY POINTS FOR ISODISTORT ANALYSIS:

1. COMPLETESS CHECK
   - Look for "Completeness test: PASS" 
   - Values >95% indicate good mode reconstruction
   - Values <90% may indicate insufficient q-point sampling

2. DOMINANT MODES
   - Top few modes typically drive the distortion
   - Look for modes with large projection coefficients
   - Pay attention to frequencies (cm⁻¹) - low frequencies often indicate soft modes

3. Q-POINT ANALYSIS
   - Γ point (0,0,0) modes often dominate ferroelectric distortions
   - Zone-boundary modes indicate more complex patterns
   - Multiple q-points suggest incommensurate or long-wavelength distortions

4. FREQUENCY SIGNS
   - Positive frequencies: stable modes
   - Negative frequencies: imaginary/soft modes (unstable)
   - Near-zero frequencies: acoustic or Goldstone modes

5. PROJECTION COEFFICIENTS
   - Large |c_qν| values: strong contribution to distortion
   - Sign indicates phase relationship
   - Squared values show relative importance

TYPICAL ISODISTORT USE CASES:

• Phase Transition Analysis
  - Identify soft modes driving structural transitions
  - Quantify mode contributions to order parameters

• Distortion Engineering  
  - Understand which phonon modes to enhance/suppress
  - Design materials with desired structural properties

• Symmetry Analysis
  - Relate distortion patterns to irreducible representations
  - Understand symmetry-lowering mechanisms

For more detailed methodology, see:
  - docs/decompose.md - Mathematical formulation
  - README.md - Command line options
""")


def main():
    """Main example execution."""

    print("ISODISTORT Analysis Example for phonproj")
    print("=" * 50)

    # Run the analysis
    success = run_isodistort_analysis()

    if success:
        # Provide interpretation guidance
        interpret_results()

        print("\n" + "=" * 80)
        print("✅ EXAMPLE COMPLETED SUCCESSFULLY")
        print("=" * 80)
        print("Try experimenting with:")
        print("• Different supercell sizes (2x2x2, 8x8x8, etc.)")
        print("• --normalize flag for relative comparisons")
        print("• --output to save results to file")
        print("• Other ISODISTORT files in data/yajundata/")

    else:
        print("\n❌ EXAMPLE FAILED")
        print("Please check the error messages above and ensure:")
        print("• phonproj is installed: uv pip install -e .")
        print("• Test data is available in data/ directory")
        print("• All dependencies are installed")

        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
