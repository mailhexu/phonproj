# Add ISODISTORT File Support to CLI

## Summary

Extend the CLI to accept ISODISTORT files as input instead of requiring displaced structure files. This enables direct analysis of ISODISTORT output files containing both undistorted and distorted structures.

## Why

Currently, users must manually extract structures from ISODISTORT files and provide them as separate displaced structure files to the CLI. This creates extra work and potential for errors. ISODISTORT files already contain both undistorted and distorted structures, making them ideal input sources for phonon mode analysis. Adding direct ISODISTORT support streamlines the workflow and reduces manual processing steps.

## What Changes

1. **CLI Argument Extension**: Add `--isodistort` argument to accept ISODISTORT files
2. **Structure Processing**: Parse ISODISTORT files to extract undistorted and distorted structures
3. **Displacement Calculation**: Compute displacement between ISODISTORT structures
4. **Structure Mapping**: Map ISODISTORT structures to phonopy supercell coordinates
5. **Pipeline Integration**: Integrate ISODISTORT workflow with existing analysis pipeline

## Files Modified

- `phonproj/cli.py`: Add ISODISTORT file parsing and displacement calculation
- `phonproj/isodistort_parser.py`: Already implemented, will be used as-is

## Key Changes

1. **New CLI argument**: `--isodistort` to accept ISODISTORT file paths
2. **Displacement calculation**: Compute displacement between undistorted and distorted structures from ISODISTORT file
3. **Structure mapping**: Map undistorted ISODISTORT structure to phonopy supercell for projection analysis
4. **Backward compatibility**: Keep existing `--displaced` argument functionality

## Test Scenario

Test with `P4mmm-ref.txt` ISODISTORT file:
- Parse undistorted and distorted structures
- Compute displacement vector
- Map to phonopy supercell from `0.02-P4mmm-PTO` directory
- Project onto phonon modes for 16x1x1 supercell
- Verify results match step 10 analysis