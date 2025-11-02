## ADDED Requirements

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