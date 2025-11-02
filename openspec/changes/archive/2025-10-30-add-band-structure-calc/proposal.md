## Why
To enable phonon band structure analysis, the project must support calculation of eigenvectors and eigendisplacements for a list of q-points, and provide a function to plot the band structure. This is required for real-world analysis and aligns with step 2 of plan.md.

## What Changes
- Implement calculation of eigenvectors and eigendisplacements for a list of q-points using the phonopy API.
- Implement a function to plot the band structure, referencing band_structure.py.
- Add tests to plot the band structure for BaTiO3 and PbTiO3.

## Impact
- Affected specs: band-structure
- Affected code: band_structure.py, core/, tests/
