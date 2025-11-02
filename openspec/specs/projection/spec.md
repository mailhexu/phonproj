# projection Specification

## Purpose
TBD - created by archiving change add-eigenvector-projection. Update Purpose after archive.
## Requirements
### Requirement: Project eigenvectors at a q-point onto an arbitrary unit vector
- The PhononModes class SHALL provide a method to project all eigenvectors at a specified q-point onto a user-supplied unit vector of matching dimension.
- The method SHALL return the projection values for each eigenvector.

#### Scenario:
- Given a q-point and a unit vector, when the projection method is called, it returns a list of projection values (one per eigenvector).

### Requirement: Verify completeness of eigenvector basis via projections
- The PhononModes class SHALL provide a method to verify that the sum of squared projections of all eigenvectors at a q-point onto a unit vector is 1 (within numerical tolerance).
- The method SHALL return a boolean indicating completeness and the computed sum.

#### Scenario:
- Given a q-point and a unit vector, when the completeness verification method is called, it returns True and a sum very close to 1 if the eigenvectors form a complete basis.

