## ADDED Requirements

### Requirement: CLI accepts ISODISTORT files as input
The CLI SHALL accept ISODISTORT files as an alternative to displaced structure files, enabling direct analysis of ISODISTORT output containing both undistorted and distorted structures.

#### Scenario: CLI accepts ISODISTORT files as input
- **WHEN** user provides `--isodistort` argument with valid ISODISTORT file path
- **THEN** the CLI SHALL successfully parse the ISODISTORT file, extract both structures, and proceed with displacement analysis

### Requirement: Compute displacement from ISODISTORT structures
The CLI SHALL compute displacement vector between undistorted and distorted structures extracted from ISODISTORT file, using the same mass-weighting and coordinate transformations as the existing displaced structure workflow.

#### Scenario: Compute displacement from ISODISTORT structures
- **WHEN** an ISODISTORT file contains undistorted and distorted PbTiO3 structures
- **THEN** the CLI SHALL compute a displacement vector that matches the physical distortion between the two structures, with proper mass-weighting applied

### Requirement: Map ISODISTORT structure to phonopy supercell
The CLI SHALL map the undistorted ISODISTORT structure to the phonopy-generated supercell, applying the same mapping to the computed displacement vector to ensure consistent coordinate systems for mode projection.

#### Scenario: Map ISODISTORT structure to phonopy supercell
- **WHEN** analyzing P4mmm-ref.txt with a 16x1x1 supercell from phonopy data
- **THEN** the CLI SHALL correctly map the ISODISTORT undistorted structure to the phonopy supercell, handling atom reordering and periodic boundary conditions

### Requirement: Maintain backward compatibility
The CLI SHALL maintain full backward compatibility with existing displaced structure input while adding ISODISTORT file support as an alternative input method.

#### Scenario: Maintain backward compatibility
- **WHEN** existing workflows use the `--displaced` argument
- **THEN** they SHALL continue to work unchanged, while new workflows can use the `--isodistort` argument for ISODISTORT files

## MODIFIED Requirements

### Requirement: CLI argument parsing
The existing CLI argument parser SHALL be modified to include the new `--isodistort` argument, making it mutually exclusive with the `--displaced` argument to prevent input ambiguity.

#### Scenario: CLI argument parsing
- **WHEN** a user attempts to provide both `--displaced` and `--isodistort` arguments
- **THEN** the CLI SHALL return a clear error message indicating that only one input type can be specified