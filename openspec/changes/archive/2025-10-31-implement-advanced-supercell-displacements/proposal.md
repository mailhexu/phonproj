## Why

The current eigendisplacement implementation uses a basic manual approach for supercell displacement calculations. To fully implement Step 7 of the project plan, we need to replace this with phonopy's advanced API capabilities that provide proper handling of commensurate q-points, bulk displacement generation, and comprehensive supercell functionality.

## What Changes

- Replace `_calculate_supercell_displacements` with phonopy API calls (`get_modulations_and_supercell`)
- Add function to generate supercell displacements for all modes at a given q-point
- Implement commensurate q-points calculation for supercells
- Add bulk displacement generation for all commensurate q-points
- Comprehensive test suite covering Gamma and non-Gamma q-points with various supercell sizes

## Impact

- Affected specs: eigendisplacement
- Affected code: `phonproj/modes.py` (core displacement calculation methods)
- Enhanced functionality: Advanced supercell displacement generation capabilities
- Testing: New comprehensive test suite for various supercell scenarios