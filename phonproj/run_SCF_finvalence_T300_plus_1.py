#!/usr/bin/env python3
#SBATCH --account=prace_id_2020225411
##SBATCH --qos=prace_id_2020225411
#SBATCH --job-name=coF
##SBATCH --exclusive
#SBATCH --time=48:00:00
#SBATCH --ntasks=48
#SBATCH --mem-per-cpu=3000 # megabytes
#SBATCH --cpus-per-task=1
#SBATCH --mail-user=jiahui.jia@uliege.be
#SBATCH --mail-type=Fail
#SBATCH --partition=cn


from ase.io import read
import numpy as np
from TB2Jflows import auto_siesta_TB2J
from ase.units import Ry
from ase import Atoms
from ase.data import atomic_numbers
import os
import sys
#os.environ["ASE_SIESTA_COMMAND"]="mpirun -n 16 siesta  < PREFIX.fdf > PREFIX.out 2> PREFIX.err"
#os.environ["ASE_SIESTA_COMMAND"]="mpirun -n 8 ~/.local/siesta_orbm/bin/siesta  < PREFIX.fdf > PREFIX.out 2> PREFIX.err"
os.system("taskset -p 0xffffffee %d"% os.getpid())


Uf={"La": 5.5,
    "Ce": 2.5,
    "Pr": 4.0,
    "Nd": 3.1,
    "Pm": 3.4,
    #"Sm": 3.3,
    "Sm": 6.0,
    "Eu": 3.0,
    "Gd": 4.6,
    "Tb": 5.0,
    "Dy": 5.0,
    "Ho": 4.9,
    "Er": 4.2,
    "Tm": 4.8,
    "Yb": 3.0,
    "Lu": 5.5
    }

Ud=8.5

Sdict={"La": 0,
    "Ce":1,
    "Pr":2,
    "Nd":3,
    "Pm":4,
    #"Sm"3,
    "Sm":5,
    "Eu":6,
    "Gd":7,
    "Tb":6,
    "Dy":5,
    "Ho":4,
    "Er":3,
    "Tm":2,
    "Yb":1,
    "Lu":0,
    }



def run_withf(Re="La", spin="spin-orbit", Bstrength=0.2, J=0.0,N=1):
    path=f"./{Re}FeO3_Mode{N}"
    #atoms = read(f"./SmFeO3_Raman_active_vibration_modes/{Re}FeO3_Mode{N}.vasp")
    atoms = read(f"../SmFeO3_Raman_active_vibration_modes/T300/plus/{Re}FeO3_Mode{N}.vasp")

    #atoms = read(f"./{Re}FeO3.vasp")
    ## replace ReFeO3 by ReGaO3
    #symbols = atoms.get_chemical_symbols()
    #new_symbols = symbols
    #for i, s in enumerate(symbols):
    #    if s =="Fe":
    #        new_symbols[i]="Ga"
    #atoms.set_chemical_symbols(new_symbols)

    m=Sdict[Re]
    ##FM
    #magmoms=[m, m, m, m, 4,-4,4,-4] + [0]*12
    ##A-AFM
    #magmoms=[m, -m, m, -m, 4,-4,4,-4] + [0]*12
    ##G
    #magmoms=[m, m, -m, -m, 4,-4,4,-4] + [0]*12
    #C-AFM
    #magmoms=[m, -m, -m, m, 4,-4,4,-4] + [0]*12

    magmoms=[4,4,-4,-4,m, -m, -m, m] + [0]*12


    # Ga spin
    #magmoms=[m, -m, -m, m, 0,0,0,0] + [0]*12


    

    atoms.set_initial_magnetic_moments(magmoms)
    synthetic_atoms={
        #"La":((5,5,5,4), (4,6,1,0)),
        Re:((5,5,5,4), (2,6,3,0))
        #elem:((5,5,5,4), (4,6,1,0))
        }

    # plus U for Fe
    Udict={"Fe":dict(n=3, l=2, U=4.5, J=0, rc=2.11)}

    # no U
    #Udict={}

    # then add U for f.
    if Re!="La":
        Udict[Re]=dict(n=4, l=3, U=Uf[Re], J=J, rc=2.5)
    auto_siesta_TB2J(path, atoms,spin=spin, elems=[f"{Re}_4f", "Fe_3d"], Udict=Udict,
            #kmesh=[7,7,5], relax=False, scf=True, TB2J=False, fincore=False, rotate_type="spin", xc="PBEsol", split_soc=True,
            kmesh=[7,5,7], relax=False, scf=True, TB2J=True, fincore=False, rotate_type="structure", xc="PBEsol", split_soc=False,
            siesta_kwargs=dict(), 
            fdf_kwargs={'SCF.DM.Tolerance':0.001, "SCF.EDM.Tolerance": "1e-1 eV", 'SCF.Mixer.Weight':'0.05', 'MaxSCFIterations':200,
                        "SCF.H.Tolerance": "1e-2 eV", "SCF.Mix.Spin": "sum+diff", 
                        "Spin.Fix":"False", "Spin.Total": m*4,
                        "DFTU.FirstIteration":False,
                        #"DM.UseSaveDM":True,
                        "DM.UseSaveDM":False,
                        "Spin.AlignZ": True,
                        "Spin.OrbitStrength": 1,
                        "MullikenInSCF":True,
                        "Bstrength": Bstrength,
                        "savehs.so": True,
                        "ProjectedDensityOfStates":["-20.00  10.00  0.200  500  eV"]
                        },
            TB2J_kwargs=dict(nproc=1, use_cache=True, Rcut=8, nz=50))



if __name__ == '__main__':
    for Bstrength in [0.7]:
        for N in range(1,6):
            try:
                run_withf(Re='Sm', spin="spin-orbit", Bstrength=Bstrength, J=0.0,N=N)
            except Exception as e:
                print(e)

