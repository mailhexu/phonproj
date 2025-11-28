#!/usr/bin/env python3
"""
Example demonstrating PhononModes functionality with real data:
(a) Print table of labels and frequencies
(b) Generate all displacements with thermal temperature at 200K

Usage:
    python example_mode_summary_and_thermal.py <path_to_phonopy_params.yaml>

This example loads phonon data from a phonopy_params.yaml file
and demonstrates the mode summary table and temperature-dependent
displacement generation capabilities.
"""

import sys
import numpy as np
from pathlib import Path
from ase.io import write as ase_write


def main():
    """
    Main function demonstrating both functionalities.
    """
    # Get YAML file path from command line or use default
    if len(sys.argv) > 1:
        yaml_file = sys.argv[1]
    else:
        # yaml_file = "/Users/hexu/projects/TmFeO3_phonon/with_relax/phonopy_params.yaml"
        yaml_file = "/Users/hexu/projects/TmFeO3_phonon/norelax/phonopy_params.yaml"

    yaml_path = Path(yaml_file)

    if not yaml_path.exists():
        print(f"ERROR: File not found: {yaml_file}")
        print(f"\nUsage: python {sys.argv[0]} <path_to_phonopy_params.yaml>")
        sys.exit(1)

    print("=" * 80)
    print("PhononModes Example: Mode Summary and Thermal Displacements")
    print("=" * 80)
    print(f"\nLoading phonon data from: {yaml_file}")
    print("Calculating phonons at Gamma point...")

    # Load phonon data for Gamma point using PhononModes.from_phonopy_yaml
    try:
        from phonproj.modes import PhononModes

        # Specify Gamma point
        qpoints = np.array([[0.0, 0.0, 0.0]])

        # Symmetry precision for force constant symmetrization
        symprec = 0.001  # Default value (more precise)

        # Load phonon modes at Gamma point with force constant symmetrization
        modes = PhononModes.from_phonopy_yaml(
            str(yaml_path), qpoints=qpoints, symprec=symprec
        )

    except Exception as e:
        print(f"\nERROR: Could not load phonon data from {yaml_file}")
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)

    print(f"\n✓ Successfully loaded phonon data")
    print(f"  - Number of atoms: {modes.n_atoms}")
    print(f"  - Number of q-points: {modes.n_qpoints}")
    print(f"  - Number of modes per q-point: {modes.n_modes}")
    print(f"  - Gauge: {modes.gauge}")
    print(f"  - Symmetry precision: {symprec}")
    print(f"  - Force constant symmetrization:")
    print(f"    • Translational & permutation symmetries (level=2)")
    print(f"    • Space group operations applied")
    print()

    # =========================================================================
    # (a) Print mode summary table with frequencies and labels
    # =========================================================================
    print("=" * 80)
    print("(a) MODE SUMMARY TABLE - Frequencies and Labels")
    print("=" * 80)
    print()

    try:
        # Use print_mode_summary_table method from modes.py
        table_output = modes.print_mode_summary_table(
            q_index=0, include_header=True, symprec=symprec
        )
        print(table_output)
        print()
        print(f"✓ Mode summary table printed successfully ({modes.n_modes} modes)")
    except Exception as e:
        print(f"✗ Error printing mode summary table: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)

    # =========================================================================
    # (b) Generate VASP structures with thermal displacements at 200K
    # =========================================================================
    print()
    print("=" * 80)
    print("(b) GENERATING VASP STRUCTURES - Thermal displacements at 200K")
    print("=" * 80)
    print()

    # Configuration
    temperature = 200.0  # Kelvin
    supercell_matrix = np.eye(3)  # 1x1x1 supercell (primitive cell)
    q_index = 0  # Gamma point

    # Create output directory
    output_dir = Path("structures")
    output_dir.mkdir(exist_ok=True)

    print(f"Configuration:")
    print(f"  - Temperature: {temperature} K")
    print(f"  - Cell: 1×1×1 (primitive cell)")
    print(f"  - Q-point index: {q_index} (Gamma point)")
    print(f"  - Output directory: {output_dir}/")
    print()

    try:
        # Use generate_modes_at_temperature method from modes.py
        thermal_displacements = modes.generate_modes_at_temperature(
            q_index=q_index, supercell_matrix=supercell_matrix, temperature=temperature
        )

        print(f"Generated thermal displacements:")
        print(f"  - Shape: {thermal_displacements.shape}")
        print(f"  - Number of modes: {thermal_displacements.shape[0]}")
        print(f"  - Number of atoms: {thermal_displacements.shape[1]}")
        print()

        # Save undisplaced structure as vasp_mode0.vasp
        undisplaced_structure = modes.primitive_cell.copy()
        undisplaced_filename = output_dir / "vasp_mode0.vasp"
        ase_write(str(undisplaced_filename), undisplaced_structure, format="vasp")
        print(f"✓ Saved undisplaced structure: {undisplaced_filename}")
        print()

        # Generate and save displaced structures
        print("Generating displaced structures:")
        print("-" * 88)
        print(
            f"{'Band':>4} {'ID':>4} {'Sign':>5} {'Freq(THz)':>10} {'Freq(cm⁻¹)':>12} {'Max Amp(Å)':>12} {'File':>25}"
        )
        print("-" * 88)

        structure_count = 0
        for mode_idx in range(thermal_displacements.shape[0]):
            # Get frequency for this mode
            freq_thz = modes.frequencies[q_index, mode_idx]
            freq_cm1 = freq_thz * 33.35641  # Convert THz to cm⁻¹

            # Get displacement for this mode (take real part)
            mode_disp = thermal_displacements[mode_idx].real
            max_amplitude = np.max(np.abs(mode_disp))

            # Generate both positive and negative displacements
            for sign, sign_str in [(+1, "+"), (-1, "-")]:
                # Create displaced structure
                displaced_structure = modes.primitive_cell.copy()
                displaced_positions = (
                    displaced_structure.get_positions() + sign * mode_disp
                )
                displaced_structure.set_positions(displaced_positions)

                # Save VASP file with sign in filename
                structure_count += 1
                vasp_filename = output_dir / f"vasp_mode{mode_idx + 1}{sign_str}.vasp"
                ase_write(str(vasp_filename), displaced_structure, format="vasp")

                # Print summary row
                print(
                    f"{mode_idx:>4} {mode_idx + 1:>4} {sign_str:>4} {freq_thz:>10.4f} {freq_cm1:>12.2f} "
                    f"{max_amplitude:>12.6f} {vasp_filename.name:>25}"
                )

        print("-" * 88)
        print(
            f"✓ Generated {structure_count} displaced structures ({thermal_displacements.shape[0]} modes × 2 signs) "
            f"+ 1 undisplaced structure"
        )
        print()
        print("Note: 'Band' is the 0-based mode index from the mode summary table")
        print("      'ID' is the 1-based structure identifier used in filenames")
        print(f"✓ All files saved to: {output_dir.absolute()}/")

    except Exception as e:
        print(f"✗ Error generating VASP structures: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)

    # =========================================================================
    # Summary
    # =========================================================================
    print()
    print("=" * 80)
    print("DEMONSTRATION COMPLETE")
    print("=" * 80)
    print()
    print("Summary:")
    print(f"✓ (a) Mode summary table with {modes.n_modes} modes")
    print(
        f"✓ (b) {structure_count} VASP structures with thermal amplitudes at {temperature}K"
    )
    print()
    print("Output files:")
    print(f"  - {output_dir}/vasp_mode0.vasp (undisplaced)")
    print(
        f"  - {output_dir}/vasp_mode{{N}}+.vasp and vasp_mode{{N}}-.vasp for N=1 to {modes.n_modes} (displaced)"
    )


if __name__ == "__main__":
    main()
