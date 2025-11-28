#!/usr/bin/env python3
"""
Test script to diagnose frequency mismatch between mode summary table and structure generation.

Purpose:
- Compare frequencies from get_mode_summary_table() with frequencies from structure generation
- Identify where the mismatch occurs

How to run:
    python test_frequency_mismatch.py <path_to_phonopy_params.yaml>

Expected behavior:
- Should show frequencies from both methods side-by-side
- Should identify any mismatches
"""

import sys
import numpy as np
from pathlib import Path


def main():
    """Main diagnostic function."""
    # Get YAML file path
    if len(sys.argv) > 1:
        yaml_file = sys.argv[1]
    else:
        yaml_file = "/Users/hexu/projects/TmFeO3_phonon/norelax/phonopy_params.yaml"

    yaml_path = Path(yaml_file)
    if not yaml_path.exists():
        print(f"ERROR: File not found: {yaml_file}")
        sys.exit(1)

    print("=" * 80)
    print("FREQUENCY MISMATCH DIAGNOSIS")
    print("=" * 80)
    print(f"\nLoading: {yaml_file}\n")

    # Load phonon modes
    from phonproj.modes import PhononModes

    qpoints = np.array([[0.0, 0.0, 0.0]])
    symprec = 0.001

    modes = PhononModes.from_phonopy_yaml(
        str(yaml_path), qpoints=qpoints, symprec=symprec
    )

    print(f"Loaded {modes.n_modes} modes at Gamma point\n")

    # =========================================================================
    # Method 1: Get frequencies from mode summary table
    # =========================================================================
    print("Method 1: Frequencies from get_mode_summary_table()")
    print("-" * 80)

    summary = modes.get_mode_summary_table(q_index=0, symprec=symprec)

    freqs_from_summary = []
    for row in summary:
        freqs_from_summary.append(row["frequency_thz"])

    print(f"Retrieved {len(freqs_from_summary)} frequencies\n")

    # =========================================================================
    # Method 2: Get frequencies directly from modes.frequencies array
    # =========================================================================
    print("Method 2: Frequencies from modes.frequencies[q_index, mode_idx]")
    print("-" * 80)

    q_index = 0
    freqs_from_array = []
    for mode_idx in range(modes.n_modes):
        freq_thz = modes.frequencies[q_index, mode_idx]
        freqs_from_array.append(freq_thz)

    print(f"Retrieved {len(freqs_from_array)} frequencies\n")

    # =========================================================================
    # Method 3: Get frequencies using 1D indexing (what summary table uses internally)
    # =========================================================================
    print("Method 3: Frequencies from modes.frequencies[q_index] (1D slice)")
    print("-" * 80)

    freqs_from_slice = modes.frequencies[q_index]
    print(f"Retrieved {len(freqs_from_slice)} frequencies\n")

    # =========================================================================
    # Compare all three methods
    # =========================================================================
    print("=" * 80)
    print("COMPARISON")
    print("=" * 80)
    print(
        f"{'Mode':>4} {'Summary(THz)':>15} {'Array 2D(THz)':>15} {'Slice 1D(THz)':>15} {'Match?':>8}"
    )
    print("-" * 80)

    all_match = True
    for i in range(modes.n_modes):
        f1 = freqs_from_summary[i]
        f2 = freqs_from_array[i]
        f3 = freqs_from_slice[i]

        match = np.allclose([f1, f2, f3], [f1, f1, f1], rtol=1e-6)
        match_str = "✓" if match else "✗ MISMATCH"

        if not match:
            all_match = False

        print(f"{i:>4} {f1:>15.6f} {f2:>15.6f} {f3:>15.6f} {match_str:>8}")

    print("-" * 80)

    if all_match:
        print("\n✓ All frequencies match! No mismatch detected.")
        print("\nThis means the issue might be:")
        print("  1. User comparing frequencies from different runs")
        print("  2. User comparing frequencies from different q-points")
        print("  3. User looking at frequencies before/after some transformation")
    else:
        print("\n✗ MISMATCHES DETECTED!")
        print("\nDifferences found between methods. Investigating further...\n")

        # Show more details about the mismatches
        print("Detailed mismatch analysis:")
        for i in range(modes.n_modes):
            f1 = freqs_from_summary[i]
            f2 = freqs_from_array[i]
            f3 = freqs_from_slice[i]

            if not np.allclose([f1, f2, f3], [f1, f1, f1], rtol=1e-6):
                print(f"\nMode {i}:")
                print(f"  Summary:      {f1:.10f} THz")
                print(f"  Array 2D:     {f2:.10f} THz")
                print(f"  Slice 1D:     {f3:.10f} THz")
                print(f"  Diff (1-2):   {f1 - f2:.10e} THz")
                print(f"  Diff (1-3):   {f1 - f3:.10e} THz")
                print(f"  Diff (2-3):   {f2 - f3:.10e} THz")

    # =========================================================================
    # Check what _run_irreps_analysis uses
    # =========================================================================
    print("\n" + "=" * 80)
    print("CHECKING INTERNAL _run_irreps_analysis")
    print("=" * 80)

    # Access internal frequency data used by irreps
    internal_freqs = modes.frequencies[
        0
    ]  # This is what _run_irreps_analysis uses at line 857

    print(f"\nFrequencies passed to _run_irreps_analysis (line 857):")
    print(f"  Type: {type(internal_freqs)}")
    print(f"  Shape: {internal_freqs.shape}")
    print(f"  First 5 values: {internal_freqs[:5]}")


if __name__ == "__main__":
    main()
