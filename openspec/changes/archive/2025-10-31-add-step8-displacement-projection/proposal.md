## Why

Step 8 of the project plan requires implementing projection of displacements between two different supercells that may have different atom ordering and positions due to periodic boundary conditions. While the underlying structure analysis functions exist, the specific displacement-to-displacement projection functionality and comprehensive tests are missing.

## What Changes

- Add a new function `project_displacements_between_supercells()` for displacement-to-displacement projection between different supercells
- Add a convenience method to the `PhononModes` class that wraps the new function
- Implement comprehensive tests covering all Step 8 scenarios:
  - Two identical supercells and displacements (normalized/unnormalized projection)
  - One supercell translated version of the other  
  - One supercell with atoms shuffled and displacements shuffled accordingly
  - Combination of translation and shuffling
- Add documentation and examples for the new functionality

## Impact

- Affected specs: eigendisplacement (adds new displacement projection requirements)
- Affected code: 
  - `phonproj/core/structure_analysis.py` (new function)
  - `phonproj/modes.py` (new method)
  - `tests/test_eigendisplacements/` (new test file for Step 8)
  - `examples/` (new example demonstrating cross-supercell projection)