# Phonon Displacement Generator

The Phonon Displacement Generator is a standalone tool for generating atomic displacements from phonon modes and saving supercell structures in VASP format.

## Overview

This tool provides both a Python API and command-line interface for:

- Loading phonopy calculation data
- Calculating phonon modes for arbitrary supercell sizes
- Generating atomic displacement patterns for specific phonon modes
- Exporting displaced supercell structures in VASP format
- Printing displacement information for analysis

## Features

- **Flexible Supercell Support**: Works with any supercell size (2×2×2, 4×1×1, 3×3×3, etc.)
- **Mode Selection**: Generate displacements for specific q-points and phonon modes
- **Configurable Amplitude**: Control the magnitude of atomic displacements
- **VASP Output**: Save structures in standard VASP format
- **Batch Processing**: Generate all possible mode structures automatically
- **CLI and API**: Use from command line or as Python library

## Installation

The displacement generator is included in the `phonproj` package. After installation:

```bash
# CLI tool available as:
phonproj-displacement --help

# Python API available as:
from phonproj.displacement import PhononDisplacementGenerator
```

## Quick Start

### Command Line Usage

```bash
# Basic usage - generate all structures for 2×2×2 supercell
phonproj-displacement -p data/phonopy.yaml -s "2 2 2" --save-dir structures/

# Print displacements without saving files
phonproj-displacement -p data/phonopy.yaml -s "2 2 2" --print-displacements

# Custom amplitude and limited output
phonproj-displacement -p data/phonopy.yaml -s "2 2 2" \
    --amplitude 0.05 --max-atoms 5 --print-displacements
```

### Python API Usage

```python
from phonproj.displacement import PhononDisplacementGenerator
import numpy as np

# Initialize generator
generator = PhononDisplacementGenerator("data/phonopy.yaml")

# Define supercell
supercell_matrix = np.diag([2, 2, 2])

# Calculate modes
modes = generator.calculate_modes(supercell_matrix)

# Generate displacement for specific mode
displacement = generator.generate_displacement(
    q_idx=0, mode_idx=0, supercell_matrix=supercell_matrix, amplitude=0.1
)

# Save structure
filename = generator.save_structure(
    q_idx=0, mode_idx=0, supercell_matrix=supercell_matrix, 
    amplitude=0.1, output_dir="structures/"
)

# Save all structures
summary = generator.save_all_structures(
    supercell_matrix=supercell_matrix, 
    output_dir="all_structures/",
    amplitude=0.1
)
```

## Command Line Interface

### Arguments

- `-p, --phonopy`: Path to phonopy YAML file (required)
- `-s, --supercell`: Supercell dimensions (e.g., "2 2 2") (required)
- `--print-displacements`: Print displacement patterns to console
- `--save-dir`: Directory to save VASP files
- `--amplitude`: Displacement amplitude (default: 0.1)
- `--max-atoms`: Maximum number of atoms to display in output
- `--quiet`: Suppress verbose output

### Examples

```bash
# Generate all 2×2×2 structures
phonproj-displacement -p BaTiO3_phonopy_params.yaml -s "2 2 2" \
    --save-dir batio3_structures/

# Show displacements for Γ-point modes
phonproj-displacement -p BaTiO3_phonopy_params.yaml -s "2 2 2" \
    --print-displacements --max-atoms 10

# Generate 4×1×1 supercell with small amplitude
phonproj-displacement -p BaTiO3_phonopy_params.yaml -s "4 1 1" \
    --amplitude 0.05 --save-dir anisotropic_structures/
```

## Python API Reference

### PhononDisplacementGenerator Class

#### Constructor

```python
PhononDisplacementGenerator(phonopy_path: str)
```

**Parameters:**
- `phonopy_path`: Path to phonopy YAML file

#### Methods

##### calculate_modes()

```python
calculate_modes(supercell_matrix: np.ndarray) -> PhononModes
```

Calculate phonon modes for the given supercell.

**Parameters:**
- `supercell_matrix`: 3×3 supercell transformation matrix

**Returns:**
- `PhononModes` object with calculated modes

##### get_commensurate_qpoints()

```python
get_commensurate_qpoints(supercell_matrix: np.ndarray) -> List[int]
```

Get indices of q-points commensurate with the supercell.

**Parameters:**
- `supercell_matrix`: 3×3 supercell transformation matrix

**Returns:**
- List of q-point indices

##### generate_displacement()

```python
generate_displacement(
    q_idx: int, 
    mode_idx: int, 
    supercell_matrix: np.ndarray, 
    amplitude: float = 0.1
) -> np.ndarray
```

Generate atomic displacement pattern for a specific phonon mode.

**Parameters:**
- `q_idx`: Q-point index
- `mode_idx`: Mode index (0-based)
- `supercell_matrix`: 3×3 supercell transformation matrix
- `amplitude`: Displacement amplitude

**Returns:**
- Displacement array with shape (n_atoms, 3)

##### save_structure()

```python
save_structure(
    q_idx: int,
    mode_idx: int, 
    supercell_matrix: np.ndarray,
    amplitude: float = 0.1,
    output_dir: str = "."
) -> str
```

Save a single displaced structure to VASP format.

**Parameters:**
- `q_idx`: Q-point index
- `mode_idx`: Mode index
- `supercell_matrix`: 3×3 supercell transformation matrix
- `amplitude`: Displacement amplitude
- `output_dir`: Directory to save file

**Returns:**
- Path to saved VASP file

##### save_all_structures()

```python
save_all_structures(
    supercell_matrix: np.ndarray,
    output_dir: str, 
    amplitude: float = 0.1
) -> Dict[str, Any]
```

Save all displaced structures for the supercell.

**Parameters:**
- `supercell_matrix`: 3×3 supercell transformation matrix
- `output_dir`: Directory to save files
- `amplitude`: Displacement amplitude

**Returns:**
- Dictionary with summary information

## Output Files

### VASP Structure Files

Structures are saved with descriptive filenames:

```
mode_q{q_idx}_m{mode_idx}_freq_{frequency:.2f}THz.vasp
```

Example:
- `mode_q0_m0_freq_-6.05THz.vasp` - Γ-point, mode 0, -6.05 THz
- `mode_q1_m2_freq_3.32THz.vasp` - Q-point 1, mode 2, 3.32 THz

### Displacement Output

When using `--print-displacements`, the output shows:

```
=== Supercell Displacements (amplitude = 0.1) ===
Found 8 commensurate q-points:
Q-point 0: [0.000, 0.000, 0.000]
--------------------------------------------------
Mode  0 (freq =    -6.05 cm⁻¹):
  Atom  0: ( -0.0000,  -0.0100,   0.0039) Å
  Atom  1: ( -0.0000,  -0.0100,   0.0039) Å
  ...
```

## Example Workflow

See `examples/phonon_displacement_generator_example.py` for a complete example:

```bash
# Run the example
uv run python examples/phonon_displacement_generator_example.py
```

This example demonstrates:
- Loading BaTiO3 phonopy data
- Calculating modes for 2×2×2 supercell
- Generating individual mode displacements
- Saving structures in VASP format
- Using both API and CLI interfaces

## Data Requirements

The tool requires phonopy calculation data in YAML format, typically containing:

- Phonopy object with force constants
- Primitive cell structure
- Unit cell structure
- Supercell information

The YAML file should be generated by phonopy and include all necessary calculation data for phonon interpolation.

## Performance Considerations

- **Memory Usage**: Scales with supercell size (n_atoms × 3 displacement components)
- **Computation Time**: Phonon calculation dominates for large supercells
- **File I/O**: VASP file writing is fast but consider disk space for many structures

For very large supercells (3×3×3 and above), consider:
- Using `--max-atoms` to limit output display
- Processing subsets of modes at a time
- Ensuring sufficient disk space for output files

## Troubleshooting

### Common Issues

1. **"YAML file not found"**: Check phonopy file path and permissions
2. **"No commensurate q-points"**: Supercell may be incompatible with phonopy data
3. **Memory errors**: Reduce supercell size or process fewer modes
4. **Permission errors**: Check write permissions for output directory

### Tips

- Start with small supercells (2×2×2) to test
- Use `--print-displacements` before saving large numbers of files
- Check phonopy data integrity with phonopy tools first
- Monitor disk space when generating many structures

## Integration with Other Tools

The generated VASP files can be used directly with:

- **VASP**: For DFT calculations of displaced structures
- **ASE**: For further structure manipulation and analysis
- **pymatgen**: For structure analysis and conversion
- **phonopy**: For additional phonon calculations

The displacement arrays can be used with other phonproj tools for:
- Mode analysis and visualization
- Projection calculations
- Structure mapping
- Eigendisplacement analysis