# phonopy-loader Specification

## Purpose
TBD - created by archiving change add-phonopy-loader. Update Purpose after archive.
## Requirements
### Requirement: Phonopy Loader
The system SHALL provide a loader function to obtain a phonopy object using phonopy.load, supporting both yaml file and directory inputs.

#### Scenario: Load phonopy object from yaml file
- **WHEN** a valid phonopy yaml file is provided
- **THEN** the loader SHALL return a phonopy object

#### Scenario: Load phonopy object from directory
- **WHEN** a valid phonopy directory is provided
- **THEN** the loader SHALL return a phonopy object

#### Scenario: Loader tested on BaTiO3 and PbTiO3
- **WHEN** tests are run for BaTiO3 (yaml) and PbTiO3 (directory)
- **THEN** the loader SHALL pass all tests and return correct objects

