## Why
The project needs a robust loader for phonopy files to obtain the phonopy object, supporting both yaml file and directory inputs. This is foundational for all subsequent phonon calculations and analysis, and aligns with step 1 of plan.md.

## What Changes
- Implement a loader for phonopy files using phonopy.load, supporting both yaml file and directory inputs.
- Reference and adapt code from refs/phonon_utils.py as needed.
- Add tests for both BaTiO3 (yaml) and PbTiO3 (directory) cases.

## Impact
- Affected specs: phonopy-loader
- Affected code: src/phonopy_loader.py (or similar), tests/
