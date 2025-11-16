# Comprehensive Mapping Tests Specification

## ADDED Requirements

### Requirement: Shuffle-Only Mapping Test
The system SHALL provide comprehensive testing of enhanced mapping functionality with atom shuffling only.

#### Scenario: Test shuffled structure mapping
- **WHEN** I test enhanced mapping using `ref.vasp` vs `ref.vasp` with atoms randomly shuffled
- **THEN** the mapping correctly identifies the original atom order despite the shuffle
- **AND** achieves near-zero final distances (< 0.001Å)
- **AND** generates detailed output showing the permutation indices
- **AND** validates that all atoms are correctly mapped

### Requirement: Shuffle and Translation Mapping Test  
The system SHALL test enhanced mapping with combined atom shuffling and translation transformations.

#### Scenario: Test shuffled and translated structures
- **WHEN** I test enhanced mapping using `ref.vasp` vs `ref.vasp` with atoms shuffled and translated by 1.0 in scaled positions
- **THEN** the mapping correctly handles both the permutation and translation
- **AND** identifies the optimal shift vector
- **AND** achieves minimal final distances after accounting for the translation
- **AND** validates the translation vector accuracy

### Requirement: Shuffle, Translation, and Displacement Test
The system SHALL test enhanced mapping with combined shuffling, translation, and displacement transformations.

#### Scenario: Test complex transformed structures
- **WHEN** I test enhanced mapping using `ref.vasp` vs `ref.vasp` with atoms shuffled, translated by 1.0 in scaled positions, and uniformly displaced by 0.1Å
- **THEN** the mapping handles all three transformations correctly
- **AND** correctly identifies the systematic displacement component
- **AND** reports appropriate final distances (~0.1Å)
- **AND** validates separation of translation and displacement

### Requirement: Cross-Structure Mapping Test
The system SHALL test enhanced mapping between different but related crystal structures.

#### Scenario: Test different structure mapping
- **WHEN** I test enhanced mapping between `ref.vasp` and `SM2.vasp`
- **THEN** the mapping handles structural differences while maintaining species constraints
- **AND** provides meaningful distance metrics
- **AND** generates detailed analysis of mapping quality
- **AND** validates species constraint enforcement

### Requirement: Supercell Mapping Test
The system SHALL test enhanced mapping between primitive and supercell structures.

#### Scenario: Test supercell transformation mapping
- **WHEN** I test enhanced mapping between `ref.vasp` and `supercell_undistorted.vasp`
- **THEN** the mapping handles the size difference correctly
- **AND** maps atoms appropriately between primitive and supercell structures
- **AND** provides detailed output showing the supercell transformation
- **AND** validates supercell expansion mapping accuracy

## MODIFIED Requirements

### Requirement: Test Data Integration
The system SHALL provide robust test data management with automatic file discovery and error handling.

#### Scenario: Automatic test data discovery
- **WHEN** the test suite runs
- **THEN** it automatically locates and uses test data files from `data/yajundata/` directory
- **AND** provides proper error handling if files are missing or malformed
- **AND** includes clear diagnostic messages for data issues

### Requirement: Test Output Validation
The system SHALL validate both mapping correctness and output file quality in comprehensive tests.

#### Scenario: Comprehensive test validation
- **WHEN** each mapping test completes
- **THEN** the system validates not only the mapping correctness
- **AND** validates the quality and completeness of generated output files
- **AND** ensures all required information is present in correct format
- **AND** checks output file structure and content accuracy