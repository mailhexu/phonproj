# eigendisplacement Specification

## Purpose
TBD - created by archiving change implement-eigendisplacement-calculation. Update Purpose after archive.
## Requirements
### Requirement: Eigendisplacement Extraction
The system SHALL provide a method to extract eigendisplacements from phonon eigenvectors.

#### Scenario: Extract eigendisplacement for BaTiO3
- **WHEN** I call `get_eigen_displacement(q_index=0, mode_index=5)` on BaTiO3 data
- **THEN** I receive a (5, 3) numpy array representing mass-weighted atomic displacements

### Requirement: Mass-Weighted Norm Calculation
The system SHALL calculate mass-weighted norms for displacement patterns.

#### Scenario: Verify unit norm for normalized eigendisplacements
- **WHEN** I call `mass_weighted_norm(displacement)` on a normalized eigendisplacement
- **THEN** the returned norm equals 1.0 within numerical tolerance (1e-10)

### Requirement: Mass-Weighted Inner Product
The system SHALL compute mass-weighted inner products between displacement patterns.

#### Scenario: Calculate orthogonality between different eigendisplacements
- **WHEN** I call `mass_weighted_projection(disp1, disp2)` for different eigendisplacements
- **THEN** the result is approximately 0.0 for orthogonal modes

### Requirement: Phonopy Integration
The system SHALL provide phonopy-compatible eigendisplacement generation.

#### Scenario: Generate supercell displacements using phonopy
- **WHEN** I call `generate_eigen_displacement_phonopy()` with supercell matrix
- **THEN** I receive real displacement vectors for all atoms in the supercell

### Requirement: Eigendisplacement Orthonormality Matrix Calculation
The system SHALL provide a method to calculate the full orthonormality matrix for all eigendisplacements at a q-point using mass-weighted inner products.

#### Scenario: Calculate orthonormality matrix for Gamma point
- **WHEN** I call `calculate_orthonormality_matrix(q_index=0)` for BaTiO3 Gamma point
- **THEN** I receive an NxN complex matrix where N is the number of phonon modes
- **AND** diagonal elements represent mass-weighted norms `<u_i|M|u_i>`
- **AND** off-diagonal elements represent mass-weighted projections `<u_i|M|u_j>`

### Requirement: Systematic Eigendisplacement Orthonormality Verification
The system SHALL provide a method to verify that eigendisplacements form an orthonormal basis under mass-weighted inner products.

#### Scenario: Verify orthonormality for BaTiO3 Gamma point
- **WHEN** I call `verify_eigendisplacement_orthonormality(q_index=0, tolerance=1e-10)` 
- **THEN** the method returns True if orthonormality is satisfied within tolerance
- **AND** provides detailed analysis of maximum deviations from identity matrix
- **AND** identifies specific modes with orthonormality violations if any

### Requirement: Comprehensive Orthonormality Analysis Reporting
The system SHALL provide detailed reporting of orthonormality verification results including deviation analysis and validation statistics.

#### Scenario: Generate detailed orthonormality report
- **WHEN** I call `verify_eigendisplacement_orthonormality()` with `detailed_report=True`
- **THEN** I receive a comprehensive report including orthonormality matrix, maximum errors, per-mode analysis, and validation summary
- **AND** the report identifies which specific eigendisplacement pairs violate orthogonality constraints

### Requirement: Advanced Supercell Displacement Generation
The system SHALL provide advanced supercell displacement generation using phonopy's API to replace manual calculations.

#### Scenario: Replace manual supercell calculations with phonopy API
- **WHEN** I call supercell displacement methods
- **THEN** the system uses phonopy's `get_modulations_and_supercell` instead of manual `_calculate_supercell_displacements`

### Requirement: Bulk Mode Displacement Generation
The system SHALL generate supercell displacements for all modes at a given q-point efficiently.

#### Scenario: Generate all mode displacements for q-point
- **WHEN** I call `generate_all_mode_displacements(q_point)` 
- **THEN** I receive displacement arrays for all phonon modes at that q-point
- **AND** each displacement respects mass-weighting conventions

### Requirement: Commensurate Q-Point Calculation
The system SHALL calculate all commensurate q-points for a given supercell.

#### Scenario: Get commensurate q-points for 2x2x2 supercell
- **WHEN** I call `get_commensurate_qpoints(supercell_matrix=[[2,0,0],[0,2,0],[0,0,2]])`
- **THEN** I receive all q-points commensurate with the supercell
- **AND** the q-points are properly normalized and formatted

### Requirement: Bulk Commensurate Displacement Generation  
The system SHALL generate displacement lists for all commensurate q-points of a supercell.

#### Scenario: Generate displacements for all commensurate q-points
- **WHEN** I call `generate_all_commensurate_displacements(supercell_matrix)`
- **THEN** I receive organized displacement data for every commensurate q-point
- **AND** each displacement maintains proper mass-weighting and normalization

### Requirement: Gamma Point Orthonormality Validation (1x1x1)
The system SHALL verify orthonormality for Gamma point displacements in unit supercells using mass-weighted inner products.

#### Scenario: Validate Gamma point orthonormality for 1x1x1 supercell
- **WHEN** I test Gamma q-point displacements with 1x1x1 supercell
- **THEN** all displacements are orthonormal with mass-weighted inner product
- **AND** the maximum deviation from identity matrix is within tolerance (1e-12)

### Requirement: Gamma Point Completeness Verification (1x1x1)
The system SHALL verify completeness of eigendisplacement basis for Gamma point in unit supercells.

#### Scenario: Verify completeness for Gamma point 1x1x1 supercell
- **WHEN** I project a random normalized displacement onto all Gamma eigendisplacements
- **THEN** the sum of squared projection coefficients equals 1.0 within tolerance
- **AND** this confirms the eigendisplacement basis is complete

### Requirement: Non-Gamma Point Orthogonality Validation (2x2x2)
The system SHALL verify orthogonality and proper normalization for non-Gamma q-points in larger supercells.

#### Scenario: Validate non-Gamma orthogonality for 2x2x2 supercell  
- **WHEN** I test non-Gamma q-point displacements with 2x2x2 supercell
- **THEN** displacements are orthogonal with mass-weighted inner product
- **AND** each displacement has mass-weighted norm of 1
- **AND** N equals 8 for a 2x2x2 supercell

### Requirement: Multi-Q-Point Completeness Verification (2x2x2)
The system SHALL verify completeness across all commensurate q-points for larger supercells.

#### Scenario: Verify completeness for all commensurate q-points 2x2x2 supercell
- **WHEN** I project a random normalized displacement onto all eigendisplacements from all commensurate q-points
- **THEN** the sum of squared projections equals 1.0 within tolerance  
- **AND** this confirms the combined eigendisplacement basis spans the full space

### Requirement: Cross-Supercell Displacement Projection
The system SHALL provide functionality to project a displacement from one supercell onto a displacement in another potentially different supercell, using mass-weighted inner product and handling structural differences through atom mapping.

#### Scenario: Project identical supercells
- **WHEN** projecting a displacement from one supercell to an identical supercell with the same displacement
- **THEN** the projection coefficient shall be 1.0 (normalized) or the ratio of mass-weighted norms (unnormalized)

#### Scenario: Project translated supercell
- **WHEN** projecting a displacement between supercells where one is a translated version of the other
- **THEN** the system shall automatically find atom correspondences and project correctly using periodic boundary conditions

#### Scenario: Project shuffled supercell
- **WHEN** projecting between supercells where atoms are shuffled and displacements are shuffled accordingly
- **THEN** the system shall find the optimal atom mapping and project the displacement correctly

#### Scenario: Project with combined transformations
- **WHEN** projecting between supercells with both translation and atom shuffling
- **THEN** the system shall handle both transformations and project correctly through automatic structure analysis

### Requirement: PhononModes Cross-Supercell Integration
The PhononModes class SHALL provide convenient methods for cross-supercell displacement projection that integrate with existing phonon mode functionality.

#### Scenario: Convenient projection method
- **WHEN** using PhononModes to project a displacement to another supercell
- **THEN** the method shall handle supercell generation and structure analysis automatically

#### Scenario: Mass-weighted projection options
- **WHEN** projecting displacements between supercells
- **THEN** the system shall support both normalized and unnormalized projection modes

### Requirement: Complete Mode Decomposition
The system SHALL provide functionality to decompose arbitrary displacements into contributions from all phonon modes across all commensurate q-points in a supercell.

#### Scenario: Decompose random displacement to all commensurate modes
- **WHEN** I call `decompose_displacement_to_modes(displacement, supercell_matrix)` with an arbitrary displacement vector
- **THEN** I receive a comprehensive breakdown showing projection coefficients for every phonon mode from all commensurate q-points
- **AND** the breakdown includes both raw projection coefficients and their squared values
- **AND** each entry is labeled with q-point index, mode index, and frequency information

### Requirement: Projection Coefficient Analysis
The system SHALL provide detailed analysis of projection coefficients including tabular output and statistical summaries.

#### Scenario: Generate projection coefficient table
- **WHEN** I decompose a displacement using `decompose_displacement_to_modes()`
- **THEN** I receive a structured table with columns for:
  - Q-point index and coordinates
  - Mode index and frequency
  - Raw projection coefficient
  - Squared projection coefficient
- **AND** the table is sorted by contribution magnitude for easy analysis

### Requirement: Completeness Verification for Mode Decomposition
The system SHALL verify that the decomposition spans the complete vector space by confirming the sum of squared projections equals 1.0 for normalized displacements.

#### Scenario: Verify completeness for random normalized displacement
- **WHEN** I decompose a mass-weighted normalized random displacement
- **THEN** the sum of all squared projection coefficients equals 1.0 within numerical tolerance (1e-12)
- **AND** this confirms the phonon mode basis is complete for the supercell

### Requirement: BaTiO3 Real Data Validation
The system SHALL support testing and validation using real BaTiO3 phonon data to ensure practical applicability.

#### Scenario: Test decomposition with BaTiO3 data
- **WHEN** I use BaTiO3 phonopy data to decompose random displacements
- **THEN** all decompositions work correctly with real crystal structure and phonon modes
- **AND** completeness verification succeeds for properly normalized displacements
- **AND** results are physically meaningful and interpretable

### Requirement: Structural Transformation Robustness
The system SHALL handle displacement decomposition for supercells with shuffled atoms and correspondingly shuffled displacements.

#### Scenario: Decompose with shuffled atom order
- **WHEN** I shuffle atoms in a supercell and correspondingly shuffle the displacement vector
- **THEN** the decomposition automatically handles the structural transformation
- **AND** projection coefficients remain unchanged from the unshuffled case
- **AND** completeness verification still succeeds

### Requirement: Integration with PhononModes Class
The PhononModes class SHALL provide convenient methods for complete mode decomposition that integrate seamlessly with existing displacement analysis functionality.

#### Scenario: Convenient decomposition through PhononModes
- **WHEN** I use PhononModes to decompose displacements
- **THEN** the system automatically handles supercell generation and mode calculation
- **AND** provides easy access to decomposition results and analysis
- **AND** supports both tabular output and programmatic access to coefficients

### Requirement: Performance and Memory Efficiency
The system SHALL handle mode decomposition efficiently for realistic supercell sizes without excessive computational or memory overhead.

#### Scenario: Efficient decomposition for large supercells
- **WHEN** decomposing displacements for supercells with hundreds of atoms
- **THEN** the computation completes in reasonable time (< 60 seconds for 2x2x2 supercells)
- **AND** memory usage remains manageable (< 1GB for typical cases)
- **AND** results maintain numerical accuracy despite computational complexity

