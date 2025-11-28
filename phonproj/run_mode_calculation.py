#!/usr/bin/env python3
"""
Run DFT calculation for a single phonon mode displacement structure.

This script reads a structure file and performs a SIESTA+TB2J calculation
in a designated directory (e.g., mode1+, mode1-, etc.).

Based on run_SCF_finvalence_T300_plus_1.py

Usage:
    python run_mode_calculation.py <structure_file> <output_dir>

Example:
    python run_mode_calculation.py structures/vasp_mode1+.vasp mode1+

SLURM directives should be provided via sbatch command or wrapper script.
"""

import sys
import os
from pathlib import Path
from ase.io import read
import numpy as np
from TB2Jflows import auto_siesta_TB2J
from ase.units import Ry


# U parameters for rare earth f-electrons (from run_SCF_finvalence_T300_plus_1.py)
Uf = {
    "La": 5.5,
    "Ce": 2.5,
    "Pr": 4.0,
    "Nd": 3.1,
    "Pm": 3.4,
    "Sm": 6.0,
    "Eu": 3.0,
    "Gd": 4.6,
    "Tb": 5.0,
    "Dy": 5.0,
    "Ho": 4.9,
    "Er": 4.2,
    "Tm": 4.8,
    "Yb": 3.0,
    "Lu": 5.5,
}

# Spin values for rare earth elements (from run_SCF_finvalence_T300_plus_1.py)
Sdict = {
    "La": 0,
    "Ce": 1,
    "Pr": 2,
    "Nd": 3,
    "Pm": 4,
    "Sm": 5,
    "Eu": 6,
    "Gd": 7,
    "Tb": 6,
    "Dy": 5,
    "Ho": 4,
    "Er": 3,
    "Tm": 2,
    "Yb": 1,
    "Lu": 0,
}


def run_siesta_calculation(
    structure_file, output_dir, Re="Tm", spin="spin-orbit", Bstrength=0.7, J=0.0
):
    """
    Run SIESTA+TB2J calculation.

    Args:
        structure_file: Path to input structure file
        output_dir: Directory where calculation runs
        Re: Rare earth element symbol (default: Tm for TmFeO3)
        spin: Spin treatment ("spin-orbit" or "non-collinear")
        Bstrength: Magnetic field strength
        J: J parameter for DFT+U
    """
    # Set thread affinity (from reference script)
    os.system("taskset -p 0xffffffee %d" % os.getpid())

    # Read structure
    print(f"Reading structure from: {structure_file}")
    atoms = read(structure_file)

    # Detect rare earth element from structure
    symbols = atoms.get_chemical_symbols()
    detected_re = None
    for symbol in set(symbols):
        if symbol in Sdict:
            detected_re = symbol
            break

    if detected_re:
        Re = detected_re
        print(f"Detected rare earth element: {Re}")
    else:
        print(f"No rare earth detected, using specified element: {Re}")

    # Get magnetic moment for rare earth
    m = Sdict.get(Re, 0)

    # Set up magnetic moments (C-AFM configuration from reference)
    # Assumes structure: 4xFe, 4xRe, 12xO (total 20 atoms)
    magmoms = [4, 4, -4, -4, m, -m, -m, m] + [0] * 12

    # Adjust if structure has different number of atoms
    if len(atoms) != len(magmoms):
        print(f"Warning: Structure has {len(atoms)} atoms, expected 20")
        print(f"Adjusting magnetic moments accordingly")
        # Simple adjustment: extend or truncate
        if len(atoms) > len(magmoms):
            magmoms.extend([0] * (len(atoms) - len(magmoms)))
        else:
            magmoms = magmoms[: len(atoms)]

    atoms.set_initial_magnetic_moments(magmoms)

    # Synthetic atoms configuration
    synthetic_atoms = {Re: ((5, 5, 5, 4), (2, 6, 3, 0))}

    # DFT+U parameters
    Udict = {"Fe": dict(n=3, l=2, U=4.5, J=0, rc=2.11)}

    # Add U for f-electrons if not La
    if Re != "La" and Re in Uf:
        Udict[Re] = dict(n=4, l=3, U=Uf[Re], J=J, rc=2.5)

    print(f"\nRunning SIESTA+TB2J calculation in {output_dir}")
    print(f"Rare earth element: {Re}")
    print(f"Spin treatment: {spin}")
    print(f"Magnetic configuration: C-AFM")
    print(f"DFT+U parameters: {Udict}")

    try:
        # Run auto_siesta_TB2J (from reference script)
        auto_siesta_TB2J(
            output_dir,
            atoms,
            spin=spin,
            elems=[f"{Re}_4f", "Fe_3d"],
            Udict=Udict,
            kmesh=[7, 5, 7],  # From reference
            relax=False,
            scf=True,
            TB2J=True,
            fincore=False,
            rotate_type="structure",
            xc="PBEsol",
            split_soc=False,
            siesta_kwargs=dict(),
            fdf_kwargs={
                "SCF.DM.Tolerance": 0.001,
                "SCF.EDM.Tolerance": "1e-1 eV",
                "SCF.Mixer.Weight": "0.05",
                "MaxSCFIterations": 200,
                "SCF.H.Tolerance": "1e-2 eV",
                "SCF.Mix.Spin": "sum+diff",
                "Spin.Fix": "False",
                "Spin.Total": m * 4,
                "DFTU.FirstIteration": False,
                "DM.UseSaveDM": False,
                "Spin.AlignZ": True,
                "Spin.OrbitStrength": 1,
                "MullikenInSCF": True,
                "Bstrength": Bstrength,
                "savehs.so": True,
                "ProjectedDensityOfStates": ["-20.00  10.00  0.200  500  eV"],
            },
            TB2J_kwargs=dict(nproc=1, use_cache=True, Rcut=8, nz=50),
        )

        print(f"\nCalculation completed successfully!")

        # Write summary
        summary_file = Path(output_dir) / "calculation_summary.txt"
        with open(summary_file, "w") as f:
            f.write(f"Structure file: {structure_file}\n")
            f.write(f"Rare earth element: {Re}\n")
            f.write(f"Number of atoms: {len(atoms)}\n")
            f.write(f"Calculation type: SIESTA+TB2J\n")
            f.write(f"Spin treatment: {spin}\n")
            f.write(f"Completed successfully\n")

        return True

    except Exception as e:
        print(f"\nCalculation failed with error:")
        print(str(e))
        import traceback

        traceback.print_exc()
        return False


def main():
    """Main function."""
    if len(sys.argv) < 3:
        print("ERROR: Incorrect number of arguments")
        print(
            f"Usage: {sys.argv[0]} <structure_file> <output_dir> [Re] [spin] [Bstrength] [J]"
        )
        print(
            f"\nExample: {sys.argv[0]} structures/vasp_mode1+.vasp mode1+ Tm spin-orbit 0.7 0.0"
        )
        print(f"\nOptional arguments:")
        print(f"  Re: Rare earth element (default: auto-detect or Tm)")
        print(f"  spin: Spin treatment (default: spin-orbit)")
        print(f"  Bstrength: Magnetic field strength (default: 0.7)")
        print(f"  J: J parameter for DFT+U (default: 0.0)")
        sys.exit(1)

    structure_file = sys.argv[1]
    output_dir = sys.argv[2]
    Re = sys.argv[3] if len(sys.argv) > 3 else "Tm"
    spin = sys.argv[4] if len(sys.argv) > 4 else "spin-orbit"
    Bstrength = float(sys.argv[5]) if len(sys.argv) > 5 else 0.7
    J = float(sys.argv[6]) if len(sys.argv) > 6 else 0.0

    # Validate inputs
    if not Path(structure_file).exists():
        print(f"ERROR: Structure file not found: {structure_file}")
        sys.exit(1)

    print("=" * 80)
    print("PHONON MODE CALCULATION (SIESTA+TB2J)")
    print("=" * 80)
    print(f"Structure file: {structure_file}")
    print(f"Output directory: {output_dir}")
    print(f"Rare earth element: {Re}")
    print(f"Spin treatment: {spin}")
    print(f"B-field strength: {Bstrength}")
    print(f"J parameter: {J}")
    print("=" * 80)
    print()

    # Run SIESTA calculation
    success = run_siesta_calculation(structure_file, output_dir, Re, spin, Bstrength, J)

    if success:
        print(f"\n✓ Calculation completed successfully")
        print(f"✓ Results saved to: {output_dir}/")
        sys.exit(0)
    else:
        print(f"\n✗ Calculation failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
