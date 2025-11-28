# Example: Mode Summary and Thermal Displacements

This example demonstrates two key features of the `PhononModes` class:

## Features

### (a) Mode Summary Table
Prints a formatted table showing:
- Mode indices
- Frequencies (THz and cm⁻¹)
- Irreducible representation labels
- Point group symmetry

### (b) Thermal Displacement Generation
Generates displaced structures with temperature-dependent amplitudes at 200K:
- Uses Bose-Einstein statistics
- Calculates thermal amplitudes for each mode
- Saves VASP-format structure files

## Usage

```bash
python example_mode_summary_and_thermal.py <path_to_phonopy_params.yaml>
```

Or use the default path:
```bash
python example_mode_summary_and_thermal.py
```

## Input Requirements

- A valid `phonopy_params.yaml` file from a phonopy calculation
- The file must contain:
  - Crystal structure (primitive cell)
  - Force constants
  - Atomic masses

## Output

### Console Output
1. Mode summary table with frequencies and labels
2. List of generated structures with amplitudes

### Files Generated
All files are saved to `structures/` directory:

- `vasp_mode0.vasp` - Undisplaced reference structure
- `vasp_mode1.vasp` to `vasp_modeN.vasp` - Displaced structures

Where N is the total number of phonon modes (3 × number of atoms).

## Configuration

You can modify these parameters in the script:

```python
temperature = 200.0          # Temperature in Kelvin
supercell_matrix = np.eye(3) # 1×1×1 supercell (primitive cell)
q_index = 0                  # Gamma point
```

## Methods Used

This example uses only existing methods from `modes.py`:

1. `PhononModes.from_phonopy_yaml(yaml_file, qpoints)` - Load phonon data
2. `modes.print_mode_summary_table(q_index)` - Print mode table
3. `modes.generate_modes_at_temperature(q_index, supercell_matrix, temperature)` - Generate thermal displacements

## Example Output

```
================================================================================
(a) MODE SUMMARY TABLE - Frequencies and Labels
================================================================================

q-point: [0.0000, 0.0000, 0.0000]
Point group: Pnma

# qx      qy      qz      band  freq(THz)   freq(cm-1)   label        IR  Raman
 0.0000  0.0000  0.0000     0      0.0000        0.00  Γ1           .    .  
 0.0000  0.0000  0.0000     1      0.0000        0.00  Γ2           .    .  
 0.0000  0.0000  0.0000     2      0.0000        0.00  Γ3           .    .  
...

================================================================================
(b) GENERATING VASP STRUCTURES - Thermal displacements at 200K
================================================================================

 ID Freq(THz)   Freq(cm⁻¹)  Max Amp(Å)                File
--------------------------------------------------------------------------------
  1     0.0000        0.00     0.000000      vasp_mode1.vasp
  2     0.0000        0.00     0.000000      vasp_mode2.vasp
  3     0.0000        0.00     0.000000      vasp_mode3.vasp
...
```

## Notes

- The example uses Gamma point (q = [0, 0, 0]) only
- All modes are generated, including acoustic modes (zero frequency)
- Thermal amplitudes scale with temperature via Bose-Einstein distribution
- VASP files use Cartesian coordinates
- The numbering convention: mode0 = undisplaced, mode1-N = displaced structures
