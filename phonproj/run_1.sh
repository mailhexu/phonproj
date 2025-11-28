#!/bin/bash
# Submission script for Nic5
#SBATCH --job-name=SFO
#SBATCH --time=2-00:00:00 # days-hh:mm:ss
#
#SBATCH --ntasks=24
#SBATCH --mem-per-cpu=3800 # megabytes
#SBATCH --partition=batch
#
#SBATCH --mail-user=jiahui.jia@uliege.be
#SBATCH --mail-type=Fail

#module --force purge
mkdir -p $GLOBALSCRATCH/$SLURM_JOB_ID


#module load releases/2022a
#module load imkl
#module load netCDF-Fortran/4.6.0-gompi-2022a
#module load Python/3.10.4-GCCcore-11.3.0-bare
#module load libxc/5.2.3-GCC-11.3.0
#
#module load FFTW.MPI/3.3.10-gompi-2022a
#module load OpenMPI/4.1.4-GCC-11.3.0
#module load OpenBLAS/0.3.20-GCC-11.3.0

#export LD_LIBRARY_PATH=/home/ulg/phythema/hexu/.local/src/wannier90-3.1.0:$LD_LIBRARY_PATH


#mpirun siesta  < siesta.fdf > siesta.out 2> siesta.err"
python run_SCF_finvalence_T300_plus_1.py


