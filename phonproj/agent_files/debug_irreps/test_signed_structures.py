"""
Test generation of structures with +/- displacements.

Purpose:
    Demonstrate that the example script now generates two structures per mode:
    one with positive displacement (+) and one with negative displacement (-).

How to run:
    This is a documentation script showing the expected behavior.
    To actually generate structures, run:

    uv run python example_mode_summary_and_thermal.py <path_to_phonopy_yaml>

Expected output:
    For N phonon modes, this generates:
    - 1 undisplaced structure: vasp_mode0.vasp
    - 2N displaced structures:
      * vasp_mode1+.vasp, vasp_mode1-.vasp
      * vasp_mode2+.vasp, vasp_mode2-.vasp
      * ...
      * vasp_modeN+.vasp, vasp_modeN-.vasp

Example output table:
    ================================================================================
    Generating displaced structures:
    --------------------------------------------------------------------------------
     ID  Sign  Freq(THz)   Freq(cm⁻¹)  Max Amp(Å)                      File
    --------------------------------------------------------------------------------
      1     +     -5.2588      -175.43     0.123456    vasp_mode1+.vasp
      1     -     -5.2588      -175.43     0.123456    vasp_mode1-.vasp
      2     +     -5.2588      -175.43     0.123456    vasp_mode2+.vasp
      2     -     -5.2588      -175.43     0.123456    vasp_mode2-.vasp
      3     +     -3.9200      -130.76     0.098765    vasp_mode3+.vasp
      3     -     -3.9200      -130.76     0.098765    vasp_mode3-.vasp
      ...

Physics explanation:
    For each phonon mode, the atomic displacements can be in two opposite directions:

    - Positive (+): atoms displaced along +eigenvector direction
    - Negative (-): atoms displaced along -eigenvector direction

    Both directions are physically equivalent since phonon eigenvectors have an
    arbitrary phase. Generating both is useful for:

    1. Visualization: See the mode oscillation in both directions
    2. DFT calculations: Use both to compute properties symmetrically
    3. Validation: Check that properties are symmetric under sign change
"""

print(__doc__)

# Example showing the loop structure
import numpy as np

print("\n" + "=" * 80)
print("Code Structure for +/- Generation")
print("=" * 80)

code = """
# For each mode, generate both +/- displacements
structure_count = 0
for mode_idx in range(n_modes):
    # Get the displacement pattern for this mode
    mode_disp = thermal_displacements[mode_idx].real
    
    # Generate both positive and negative structures
    for sign, sign_str in [(+1, "+"), (-1, "-")]:
        # Apply signed displacement
        displaced_positions = equilibrium_positions + sign * mode_disp
        
        # Save with sign in filename
        filename = f"vasp_mode{mode_idx + 1}{sign_str}.vasp"
        save_structure(displaced_positions, filename)
        structure_count += 1

# Total: 2 × n_modes structures (plus 1 undisplaced)
"""

print(code)

print("\n" + "=" * 80)
print("File Naming Convention")
print("=" * 80)
print("""
Old behavior:
  vasp_mode0.vasp  (undisplaced)
  vasp_mode1.vasp  (mode 1, positive only)
  vasp_mode2.vasp  (mode 2, positive only)
  ...

New behavior:
  vasp_mode0.vasp  (undisplaced)
  vasp_mode1+.vasp (mode 1, positive)
  vasp_mode1-.vasp (mode 1, negative)
  vasp_mode2+.vasp (mode 2, positive)
  vasp_mode2-.vasp (mode 2, negative)
  ...

Total structures:
  Old: 1 + N = N+1 structures
  New: 1 + 2N = 2N+1 structures
""")
