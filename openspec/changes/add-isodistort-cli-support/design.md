# Design Document: ISODISTORT CLI Integration

## Overview

This design extends the existing CLI to support ISODISTORT files as input, enabling direct analysis of ISODISTORT output files containing both undistorted and distorted structures.

## Architecture

### Current CLI Flow
1. Load phonopy data (directory or YAML file)
2. Load displaced structure (VASP format)
3. Generate reference supercell from phonopy primitive cell
4. Compute displacement between displaced and reference structures
5. Project displacement onto phonon modes

### New CLI Flow with ISODISTORT
1. Load phonopy data (directory or YAML file)
2. Load ISODISTORT file and extract structures
3. Compute displacement between undistorted and distorted ISODISTORT structures
4. Map undistorted ISODISTORT structure to phonopy supercell
5. Apply same mapping to computed displacement vector
6. Project displacement onto phonon modes

## Key Design Decisions

### 1. Mutually Exclusive Input Arguments
- `--displaced` and `--isodistort` arguments are mutually exclusive
- Prevents ambiguity in input specification
- Clear error messages guide users to correct usage

### 2. Structure Mapping Strategy
- Use existing `create_atom_mapping` function from `structure_analysis.py`
- Map undistorted ISODISTORT structure to phonopy supercell
- Apply same mapping to displacement vector
- Ensures consistent coordinate systems for mode projection

### 3. Displacement Calculation
- Compute displacement in ISODISTORT coordinate system first
- Apply structure mapping to displacement vector
- Use same mass-weighting as existing workflow
- Maintains consistency with current analysis methods

### 4. Integration Points
- Leverage existing `isodistort_parser.py` module
- Reuse `calculate_displacement_vector` function with ISODISTORT structures
- Integrate with existing `analyze_displacement` pipeline
- Minimal changes to core analysis logic

## Implementation Strategy

### Phase 1: CLI Argument Updates
- Add `--isodistort` argument to argument parser
- Implement mutual exclusion with `--displaced` argument
- Update help text and examples

### Phase 2: ISODISTORT Processing
- Add function to load and process ISODISTORT files
- Extract undistorted and distorted structures
- Compute initial displacement vector

### Phase 3: Structure Mapping
- Map undistorted ISODISTORT structure to phonopy supercell
- Apply mapping to displacement vector
- Handle periodic boundary conditions

### Phase 4: Integration and Testing
- Integrate with existing analysis pipeline
- Add comprehensive tests
- Update documentation

## Benefits

1. **Direct ISODISTORT Support**: No need to manually extract structures from ISODISTORT files
2. **Consistent Workflow**: Same analysis pipeline for both input types
3. **Backward Compatibility**: Existing workflows remain unchanged
4. **Reduced Errors**: Automated structure mapping reduces manual errors

## Risks and Mitigations

### Risk: Structure Mapping Complexity
- **Mitigation**: Use existing, tested mapping functions
- **Fallback**: Provide clear error messages for mapping failures

### Risk: Coordinate System Mismatches
- **Mitigation**: Apply consistent transformations throughout pipeline
- **Validation**: Verify mapping results with test cases

### Risk: Performance Impact
- **Mitigation**: Minimal additional computation for ISODISTORT parsing
- **Optimization**: Cache mapping results where possible