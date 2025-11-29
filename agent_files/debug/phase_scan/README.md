# Phase-Scan Bug Fix

## Problem

When using `--phase-scan`, modes that should have zero projection were showing non-zero values, while standard decomposition correctly showed zeros.

## Root Cause

The phase-scan implementation was incorrectly handling the projection of complex eigenvectors:

**Incorrect approach** (before fix):
```python
# Kept source as complex throughout
source_displacement = canonical_displacement * exp(i*phase)
coeff = project_displacements_between_supercells(
    source_displacement=source_displacement,  # Complex!
    target_displacement=target,  # Real
    normalize=False,
    use_mass_weighting=True
)
# This computes: Re[<source.conj(), target>]
# = Re[<canonical.conj() * exp(-i*phase), target>]
```

This formula doesn't correctly represent the phase-rotated physical displacement pattern.

## Solution

The correct approach is to rotate the phase in the complex plane, then take the real part (which represents the physical displacement):

```python
for phase in phases:
    # Physical displacement at this phase
    source_displacement = (canonical_displacement * exp(i*phase)).real
    
    # Project with normalization
    coeff = project_displacements_between_supercells(
        source_displacement=source_displacement,  # Real
        target_displacement=target,  # Real
        normalize=True,  # Normalize since real part may have different norm
        use_mass_weighting=True
    )
```

At phase=0: `source = Re[canonical]`, which matches standard decomposition.

**Key insight:** Standard decomposition projects `Re[canonical]` onto the target. Phase-scan should project `Re[canonical * exp(i*phase)]` for different phases.

## Files Changed

- `phonproj/core/structure_analysis.py`: Fixed `project_displacement_with_phase_scan()` to take real part before projecting
- `phonproj/cli.py`: Added displacement normalization in `analyze_phase_scan()` for consistency

## Test

Run: `uv run python agent_files/debug/phase_scan/test_phase_zero_match.py`

This verifies that phase-scan at phase=0 exactly matches standard decomposition for all modes.
