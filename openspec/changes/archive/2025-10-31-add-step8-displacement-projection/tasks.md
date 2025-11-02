## 1. Implementation

- [x] 1.1 Add `project_displacements_between_supercells()` function to `phonproj/core/structure_analysis.py`
- [x] 1.2 Add convenience method `project_displacement_to_supercell()` to `PhononModes` class
- [x] 1.3 Create comprehensive test file `tests/test_eigendisplacements/test_step8_displacement_projection.py`
- [x] 1.4 Implement all four Step 8 test scenarios:
  - [x] 1.4.1 Test identical supercells and displacements (normalized/unnormalized)
  - [x] 1.4.2 Test translated supercell scenario
  - [x] 1.4.3 Test shuffled atoms and displacements scenario  
  - [x] 1.4.4 Test combined translation and shuffling scenario
- [x] 1.5 Create example file `examples/step8_cross_supercell_projection.py`
- [x] 1.6 Update documentation for new functionality

## 2. Validation

- [x] 2.1 Run all Step 8 tests to ensure they pass
- [x] 2.2 Verify integration with existing PhononModes functionality
- [x] 2.3 Check example runs successfully
- [x] 2.4 Validate mass-weighted projection calculations are correct