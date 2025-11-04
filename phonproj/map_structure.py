#!/usr/bin/env python3
"""
Structure Mapping CLI Tool

This command-line interface provides access to enhanced structure mapping functionality
including PBC-aware distance calculations, origin alignment, shift optimization,
and detailed output generation.

Usage:
    python -m phonproj.map_structure [options] reference_structure target_structure

Examples:
    # Basic mapping with output
    python -m phonproj.map_structure ref.vasp displaced.vasp

    # Enhanced mapping with shift optimization
    python -m phonproj.map_structure ref.vasp displaced.vasp --optimize-shift --align-origin

    # Generate detailed analysis report
    python -m phonproj.map_structure ref.vasp displaced.vasp --output analysis.txt

    # Custom species mapping
    python -m phonproj.map_structure ref.vasp displaced.vasp --species-map Pb:Sr,Ti:Zr
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, Optional, Union

import numpy as np
from ase import Atoms
from ase.io import read

from phonproj.core.structure_analysis import (
    MappingAnalyzer,
    create_atom_mapping,
    create_enhanced_atom_mapping,
)


def parse_species_map(
    species_map_str: str,
) -> Optional[Dict[str, Union[str, list[str]]]]:
    """
    Parse species mapping string into dictionary.

    Format: "Sp1:Sp2,Sp3:Sp4" or "Sp1:Sp2,Sp3:Sp4,Sp5"

    Args:
        species_map_str: Species mapping string

    Returns:
        Dictionary mapping source species to target species or list of allowed species
    """
    if not species_map_str:
        return None

    species_map = {}
    for mapping in species_map_str.split(","):
        mapping = mapping.strip()
        if ":" in mapping:
            parts = mapping.split(":")
            if len(parts) != 2:
                continue  # Skip invalid mappings
            source, targets = parts[0].strip(), parts[1].strip()
            targets_list = targets.split(";") if ";" in targets else [targets]
            if len(targets_list) == 1:
                species_map[source] = targets_list[0].strip()
            else:
                species_map[source] = [t.strip() for t in targets_list]
        else:
            # Single species - allow mapping to same species only
            species_map[mapping] = mapping

    return species_map


def load_structure(filepath: str) -> Atoms:
    """
    Load structure from file with error handling.

    Args:
        filepath: Path to structure file

    Returns:
        ASE Atoms object
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Structure file not found: {filepath}")

    try:
        structure = read(filepath)
        # read() may return list or single Atoms object
        if isinstance(structure, list):
            if len(structure) == 0:
                raise ValueError(f"No structures found in file: {filepath}")
            structure = structure[0]
        return structure
    except Exception as e:
        raise ValueError(f"Error loading structure from {filepath}: {e}") from e


def print_structure_info(structure: Atoms, name: str) -> None:
    """
    Print basic information about a structure.

    Args:
        structure: ASE Atoms object
        name: Name of the structure for display
    """
    print(f"\n{name}:")
    print(f"  Atoms: {len(structure)}")
    print(f"  Species: {', '.join(set(structure.get_chemical_symbols()))}")
    print(f"  Cell: {structure.get_cell().array.diagonal()}")
    print(f"  Volume: {structure.get_volume():.2f} Ų")


def print_mapping_results(
    mapping: np.ndarray,
    cost: float,
    shift_vector: np.ndarray,
    quality: dict,
    verbose: bool = False,
) -> None:
    """
    Print mapping results in a formatted way.

    Args:
        mapping: Atom mapping array
        cost: Total mapping cost
        shift_vector: Applied shift vector
        quality: Quality metrics dictionary
        verbose: Whether to print detailed information
    """
    print("\nMapping Results:")
    print(f"  Total atoms mapped: {len(mapping)}")
    print(f"  Total cost: {cost:.6f} Å")
    print(
        f"  Shift vector: [{shift_vector[0]:.6f}, {shift_vector[1]:.6f}, {shift_vector[2]:.6f}] Å"
    )
    print(f"  Shift magnitude: {np.linalg.norm(shift_vector):.6f} Å")

    if verbose:
        print("\nQuality Metrics:")
        print(f"  Mean distance: {quality['mean_distance']:.6f} Å")
        print(f"  Max distance: {quality['max_distance']:.6f} Å")
        print(f"  Min distance: {quality['min_distance']:.6f} Å")
        print(f"  Std distance: {quality['std_distance']:.6f} Å")
        print(f"  Atoms > 0.1Å: {quality['atoms_above_01angstrom']}")
        print(f"  Atoms > 0.5Å: {quality['atoms_above_05angstrom']}")


def main() -> None:
    """
    Main CLI function for structure mapping tool.
    """
    parser = argparse.ArgumentParser(
        description="Enhanced structure mapping tool with PBC-aware distance calculations and detailed analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s ref.vasp displaced.vasp
  %(prog)s ref.vasp displaced.vasp --optimize-shift --align-origin
  %(prog)s ref.vasp displaced.vasp --output analysis.txt --verbose
  %(prog)s ref.vasp displaced.vasp --species-map Pb:Sr,Ti:Zr --no-shift-optimization
        """,
    )

    # Positional arguments
    parser.add_argument(
        "reference", help="Reference structure file (VASP POSCAR/CONTCAR format)"
    )
    parser.add_argument(
        "target", help="Target structure file (VASP POSCAR/CONTCAR format)"
    )

    # Mapping options
    parser.add_argument(
        "--method",
        choices=["distance", "enhanced"],
        default="enhanced",
        help="Mapping method to use (default: enhanced)",
    )
    parser.add_argument(
        "--species-map",
        type=str,
        help='Species mapping (e.g., "Pb:Sr,Ti:Zr" or "Pb:Sr;Ti:Zr")',
    )
    parser.add_argument(
        "--max-cost",
        type=float,
        default=1e-3,
        help="Maximum allowed total cost (default: 1e-3)",
    )

    # Enhanced mapping options
    parser.add_argument(
        "--optimize-shift",
        action="store_true",
        default=True,
        help="Enable shift vector optimization (default: enabled)",
    )
    parser.add_argument(
        "--no-shift-optimization",
        action="store_true",
        help="Disable shift vector optimization",
    )
    parser.add_argument(
        "--align-origin",
        action="store_true",
        default=True,
        help="Align structures to common origin (default: enabled)",
    )
    parser.add_argument(
        "--no-origin-alignment", action="store_true", help="Disable origin alignment"
    )
    parser.add_argument(
        "--force-near-origin",
        action="store_true",
        default=True,
        help="Force atoms near origin before mapping (default: enabled)",
    )
    parser.add_argument(
        "--no-force-near-origin", action="store_true", help="Disable force near origin"
    )

    # Output options
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        help="Output file for detailed analysis (default: auto-generated)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/mapping",
        help="Output directory for analysis files (default: data/mapping)",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Print detailed information"
    )
    parser.add_argument(
        "--quiet", "-q", action="store_true", help="Suppress non-error output"
    )

    # Utility options
    parser.add_argument(
        "--info-only",
        action="store_true",
        help="Only show structure information, no mapping",
    )
    parser.add_argument("--version", action="version", version="%(prog)s 1.0.0")

    args = parser.parse_args()

    # Handle conflicting options
    if args.no_shift_optimization:
        args.optimize_shift = False
    if args.no_origin_alignment:
        args.align_origin = False
    if args.no_force_near_origin:
        args.force_near_origin = False

    try:
        # Load structures
        if not args.quiet:
            print("Loading structures...")

        ref_structure = load_structure(args.reference)
        target_structure = load_structure(args.target)

        # Print structure information
        if not args.quiet:
            print_structure_info(ref_structure, "Reference Structure")
            print_structure_info(target_structure, "Target Structure")

        # Check structure compatibility
        if len(ref_structure) != len(target_structure):
            error_msg = f"Structures have different atom counts: {len(ref_structure)} vs {len(target_structure)}"
            if args.quiet:
                print(error_msg, file=sys.stderr)
            else:
                print(f"\nERROR: {error_msg}")
            sys.exit(1)

        # If info only, exit here
        if args.info_only:
            sys.exit(0)

        # Parse species mapping
        species_map = parse_species_map(args.species_map)

        # Perform mapping
        if not args.quiet:
            print(f"\nPerforming {args.method} mapping...")
            if species_map:
                print(f"Using species mapping: {species_map}")

        if args.method == "enhanced":
            mapping, cost, shift_vector, quality = create_enhanced_atom_mapping(
                ref_structure,
                target_structure,
                method="distance",
                max_cost=args.max_cost,
                species_map=species_map,
                optimize_shift=args.optimize_shift,
                origin_alignment=args.align_origin,
                force_near_origin=args.force_near_origin,
            )
        else:  # basic mapping
            mapping, cost = create_atom_mapping(
                ref_structure,
                target_structure,
                method="distance",
                max_cost=args.max_cost,
                species_map=species_map,
            )
            # Create dummy values for compatibility
            shift_vector = np.zeros(3)
            quality = {
                "mean_distance": cost / len(mapping),
                "max_distance": cost,
                "min_distance": 0.0,
                "std_distance": 0.0,
                "atoms_above_01angstrom": 0,
                "atoms_above_05angstrom": 0,
                "shift_magnitude": 0.0,
            }

        # Print results
        print_mapping_results(mapping, cost, shift_vector, quality, args.verbose)

        # Generate detailed output if requested
        if args.method == "enhanced":
            # Create output directory
            os.makedirs(args.output_dir, exist_ok=True)

            # Determine output filename
            if args.output:
                output_file = args.output
            else:
                # Generate automatic filename
                ref_name = Path(args.reference).stem
                target_name = Path(args.target).stem
                timestamp = (
                    __import__("datetime").datetime.now().strftime("%Y%m%d_%H%M%S")
                )
                output_file = os.path.join(
                    args.output_dir,
                    f"mapping_{ref_name}_to_{target_name}_{timestamp}.txt",
                )

            # Generate analysis
            analyzer = MappingAnalyzer(
                ref_structure, target_structure, mapping, shift_vector, quality
            )
            analyzer.save_detailed_output(output_file)

            if not args.quiet:
                print(f"\nDetailed analysis saved to: {output_file}")

        # Print mapping summary if verbose
        if args.verbose:
            print("\nMapping Summary:")
            for i, target_idx in enumerate(mapping):
                ref_species = ref_structure.get_chemical_symbols()[i]
                target_species = target_structure.get_chemical_symbols()[target_idx]
                print(
                    f"  {i:3d} -> {target_idx:3d}  {ref_species:2s} -> {target_species:2s}"
                )

    except FileNotFoundError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)
    except ValueError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nOperation cancelled by user.", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr)
        if not args.quiet:
            import traceback

            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
