## ADDED Requirements

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