# Proposal: Add Eigenvector Projection Capability

## Why
Step 4 of the project plan requires implementing eigenvector projection functionality to:
1. Project eigenvectors at a q-point onto an arbitrary unit vector
2. Verify completeness of the eigenvector basis (sum of squared projections = 1)

This capability is essential for understanding the mathematical properties of phonon modes
and verifying that the eigenvector basis is complete and orthonormal.

## What Changes
- **ADDED**: New method `project_eigenvectors()` to PhononBand class for projecting eigenvectors onto arbitrary unit vectors
- **ADDED**: New method `verify_completeness()` to test that eigenvectors form a complete basis
- **ADDED**: Tests in `tests/test_eigenvectors/` to verify projection and completeness
- **ADDED**: Example in `examples/` demonstrating eigenvector projection

## Impact
- Affected specs: `band-structure` - adds projection capabilities to existing phonon band analysis
- Affected code:
  - `phonproj/band_structure.py` - new projection methods
  - `tests/test_eigenvectors/` - new tests for projection
  - `examples/` - new example demonstrating projection
- Breaking changes: None
