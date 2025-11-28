## 1. CLI and Plumbing
- [ ] 1.1 Review existing `phonproj-decompose` CLI behavior and how it loads phonon modes and displacement vectors.
- [ ] 1.2 Design `--phase-scan` CLI options (flags and arguments) needed to identify a specific `(q_index, mode_index)` to scan.
- [ ] 1.3 Wire `--phase-scan` into `phonproj/cli.py` so the CLI branches from the normal decomposition path into a phase-scan-specific flow.

## 2. Phase-Scan Integration
- [ ] 2.1 Reuse or lightly wrap `project_displacement_with_phase_scan` and `find_maximum_projection` to operate on the CLI-computed displacement and phonon modes.
- [ ] 2.2 Ensure correct supercell generation, atom mapping, and mass-weighting assumptions are aligned between CLI and `project_displacement_with_phase_scan`.
- [ ] 2.3 Implement user-facing output for phase vs projection coefficients (and the optimal phase) in a clear tabular or summary format.

## 3. Validation
- [ ] 3.1 Add or extend tests to cover `--phase-scan` behavior using existing BaTiO3 or test fixtures where possible.
- [ ] 3.2 Manually run `phonproj-decompose` with and without `--phase-scan` on a small test case to verify that default behavior is unchanged and phase-scan outputs are sensible.
- [ ] 3.3 Update or add documentation snippets (CLI help text and examples) to mention `--phase-scan` usage.
