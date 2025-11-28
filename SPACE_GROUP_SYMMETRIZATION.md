# Space Group Symmetrization of Force Constants

## Overview

Force constants are now symmetrized using both translational/permutation symmetries and space group operations to ensure the dynamical matrix fully respects crystal symmetry.

## Implementation

### Location
`phonproj/modes.py`, line ~2768-2777

### Method
Two-step symmetrization process in `PhononModes.from_phonopy_yaml()`:

```python
if phonopy.force_constants is not None:
    # Method 1: Apply translational and permutation symmetries
    # Level >1 iteratively improves symmetrization
    phonopy.symmetrize_force_constants(level=2, show_drift=False)
    
    # Method 2: Apply space group operations (except pure translations)
    # This enforces full crystal symmetry on force constants
    phonopy.symmetrize_force_constants_by_space_group(show_drift=False)
```

## Phonopy API Methods Used

### 1. `phonopy.symmetrize_force_constants(level=2, show_drift=False)`
- Applies translational and permutation symmetries
- `level=2`: Iterates twice for better convergence
- Ensures acoustic sum rules are satisfied

### 2. `phonopy.symmetrize_force_constants_by_space_group(show_drift=False)`
- Applies all space group operations (rotations, reflections, inversions)
- Excludes pure translations (already handled by method 1)
- Enforces point group symmetry on force constants

## Benefits

1. **Exact degeneracies**: Symmetry-equivalent modes have identical frequencies
2. **Acoustic sum rule**: Translation modes have exactly zero frequency at Gamma
3. **Physical phonon spectra**: Respects crystal symmetry throughout Brillouin zone
4. **Stable calculations**: Reduces numerical noise in eigenvalue problems

## Usage

The symmetrization is automatically applied when loading phonon data:

```python
from phonproj.modes import PhononModes
import numpy as np

qpoints = np.array([[0.0, 0.0, 0.0]])  # Gamma point
modes = PhononModes.from_phonopy_yaml(
    'phonopy_params.yaml',
    qpoints=qpoints,
    symprec=0.001  # Symmetry precision
)
```

## Parameters

- `symprec` (float): Symmetry precision for detecting symmetry operations
  - Default: 0.001 Å
  - Smaller values = stricter symmetry requirements
  - Larger values = more tolerant (useful for approximate structures)

## Output

The example script confirms symmetrization:

```
✓ Successfully loaded phonon data
  - Number of atoms: 20
  - Number of q-points: 1
  - Number of modes per q-point: 60
  - Gauge: R
  - Symmetry precision: 0.001
  - Force constant symmetrization:
    • Translational & permutation symmetries (level=2)
    • Space group operations applied
```

## Verification

Check that acoustic modes have near-zero frequencies:
```python
# Gamma point modes
freqs = modes.frequencies[0]
print(f"Acoustic modes: {freqs[:3]} THz")
# Should show: ~0.0000 THz for first 3 modes
```

## References

1. Phonopy documentation: https://phonopy.github.io/phonopy/
2. Togo & Tanaka, Scr. Mater. 108, 1-5 (2015)
3. Space group theory: International Tables for Crystallography Vol. A

## Notes

- Both methods are complementary and should be used together
- Order matters: apply `symmetrize_force_constants()` first
- Force constants must be present (calculated from FORCE_SETS or dynamical matrices)
- The symmetrization respects the primitive cell used in the calculation
