# Enhanced Structure Mapping with Detailed Output

## Summary

Implement advanced structure mapping functionality that computes optimal atom correspondence between two structures considering periodic boundary conditions, with comprehensive detailed output saved to text files.

## Rationale

Current structure mapping functionality exists but lacks detailed logging and analysis capabilities. Step 12 requires enhanced mapping with:
1. PBC-optimized distance calculations
2. Origin alignment using closest atom to origin
3. Optimal mapping with shift vectors
4. Detailed output tables saved to data/mapping directory

## Proposed Changes

This change introduces enhanced structure mapping capabilities with detailed analysis output, extending existing functionality in `phonproj/core/structure_analysis.py`.

## Files to be Modified

- `phonproj/core/structure_analysis.py` - Enhanced mapping functions
- `tests/` - New test suite for mapping functionality  
- `data/mapping/` - Output directory for mapping details
- Examples and documentation updates

## Test Data

Uses existing test structures from `data/yajundata/`:
- ref.vasp vs ref.vasp (shuffled)
- ref.vasp vs ref.vasp (shuffled + translated)
- ref.vasp vs ref.vasp (shuffled + translated + displaced)
- ref.vasp vs SM2.vasp
- ref.vasp vs supercell_undistorted.vasp