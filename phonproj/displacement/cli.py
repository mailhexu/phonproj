#!/usr/bin/env python3
"""
Phonon Displacement Generator CLI

A standalone command-line tool for generating phonon mode displacements
and saving supercell structures.
"""

import argparse
import sys
import numpy as np
from pathlib import Path

from phonproj.displacement import PhononDisplacementGenerator


def parse_supercell_matrix(supercell_str: str) -> np.ndarray:
    """Parse supercell matrix from string like '2x2x2' or '2 2 2'."""
    if "x" in supercell_str:
        parts = supercell_str.split("x")
    else:
        parts = supercell_str.split()

    if len(parts) != 3:
        raise ValueError(f"Supercell must have 3 dimensions, got: {supercell_str}")

    try:
        n1, n2, n3 = map(int, parts)
    except ValueError as e:
        raise ValueError(f"Supercell dimensions must be integers: {e}")

    return np.diag([n1, n2, n3])


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Generate phonon mode displacements and save supercell structures",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Print all displacements for 2x2x2 supercell
  phonproj-displacement -p phonopy_params.yaml -s 2x2x2 --print-displacements

  # Save all structures with amplitude 0.05
  phonproj-displacement -p phonopy_params.yaml -s 2x2x2 --save-dir output --amplitude 0.05

  # Both print and save with custom amplitude
  phonproj-displacement -p phonopy_params.yaml -s 4x1x1 --print-displacements --save-dir structures --amplitude 0.1
        """,
    )

    parser.add_argument(
        "-p",
        "--phonopy",
        required=True,
        help="Path to phonopy_params.yaml file or directory containing phonopy files",
    )

    parser.add_argument(
        "-s",
        "--supercell",
        required=True,
        help="Supercell size in format NxMxL (e.g., 2x2x2, 16x1x1) or 'N M L' (e.g., '16 1 1')",
    )

    parser.add_argument(
        "--print-displacements",
        action="store_true",
        help="Print all supercell displacements for commensurate q-points",
    )

    parser.add_argument(
        "--save-dir",
        type=str,
        metavar="DIRECTORY",
        help="Save all supercell structures with displacements to specified directory in VASP format",
    )

    parser.add_argument(
        "--amplitude",
        type=float,
        default=0.1,
        help="Amplitude for displacement generation (default: 0.1)",
    )

    parser.add_argument(
        "--max-atoms",
        type=int,
        default=None,
        help="Maximum number of atoms to show per mode when printing (default: all atoms)",
    )

    parser.add_argument("--quiet", action="store_true", help="Reduce output verbosity")

    args = parser.parse_args()

    # Validate arguments
    if not args.print_displacements and not args.save_dir:
        parser.error("Must specify either --print-displacements or --save-dir")

    try:
        # Parse supercell matrix
        supercell_matrix = parse_supercell_matrix(args.supercell)

        # Initialize displacement generator
        if not args.quiet:
            print(f"Loading phonopy data from: {args.phonopy}")

        generator = PhononDisplacementGenerator(args.phonopy)

        if not args.quiet:
            n1, n2, n3 = (
                int(supercell_matrix[0, 0]),
                int(supercell_matrix[1, 1]),
                int(supercell_matrix[2, 2]),
            )
            print(f"Supercell: {n1}×{n2}×{n3}")
            print(f"Amplitude: {args.amplitude}")

        # Print displacements if requested
        if args.print_displacements:
            generator.print_displacements(
                supercell_matrix,
                amplitude=args.amplitude,
                max_atoms_per_mode=args.max_atoms,
            )

        # Save structures if requested
        if args.save_dir:
            result = generator.save_all_structures(
                supercell_matrix, output_dir=args.save_dir, amplitude=args.amplitude
            )

            if not args.quiet:
                print(f"\nSummary:")
                print(f"  Total files saved: {result['total_saved']}")
                print(f"  Output directory: {result['output_dir']}")
                print(f"  Amplitude used: {result['amplitude']}")

        if not args.quiet:
            print(f"\n✅ Displacement generation completed successfully")

    except Exception as e:
        print(f"\n❌ Error: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
