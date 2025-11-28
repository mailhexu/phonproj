## Why
Users want to analyze how a target structural displacement projects onto a given phonon mode as a function of phase, directly from the `phonproj-decompose` CLI, without having to write custom Python that calls `project_displacement_with_phase_scan` manually.

## What Changes
- Add a `--phase-scan` option to the `phonproj-decompose` CLI in `phonproj/cli.py` to request phase-resolved projection analysis instead of the standard full-mode decomposition summary.
- When `--phase-scan` is enabled, integrate with the existing `project_displacement_with_phase_scan` helper so the CLI computes and reports projection coefficients vs phase for a selected `(q_index, mode_index)` and supercell.
- Keep existing behavior as the default when `--phase-scan` is not provided to avoid breaking current workflows.

## Impact
- Affected specs: `eigendisplacement` (add requirements for CLI-accessible phase-scan projection functionality tied to existing projection helpers).
- Affected code: `phonproj/cli.py`, `phonproj/core/structure_analysis.py` and `phonproj/modes.py` usage patterns for `project_displacement_with_phase_scan`.
