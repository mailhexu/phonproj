## ADDED Requirements
### Requirement: Phase-Scan Projection via CLI
The system SHALL expose phase-scan projection of a phonon mode onto a target displacement through the `phonproj-decompose` command-line interface.

#### Scenario: Enable phase-scan from CLI
- **WHEN** I run `phonproj-decompose` with `--phase-scan` plus arguments selecting a specific `q_index` and `mode_index`
- **THEN** the tool computes projection coefficients as a function of phase using the existing `project_displacement_with_phase_scan` helper
- **AND** prints a clear summary of phases, coefficients (or their magnitudes), and the phase corresponding to the maximum projection

#### Scenario: Preserve default decomposition behavior
- **WHEN** I run `phonproj-decompose` without `--phase-scan`
- **THEN** the tool performs the existing full-mode decomposition workflow described in current specs
- **AND** no phase-scan specific output or behavior is triggered

#### Scenario: Validate phase-scan against known displacement
- **WHEN** I use `--phase-scan` on a test displacement known to align strongly with a specific phonon mode at some phase
- **THEN** the reported maximum projection coefficient is close to 1.0 (for a properly normalized displacement)
- **AND** the corresponding optimal phase is consistent with the known analytic or reference result
