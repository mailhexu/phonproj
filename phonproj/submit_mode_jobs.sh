#!/bin/bash
###############################################################################
# Submit SLURM job for all phonon mode displacement structures (SERIAL)
#
# This script creates and submits a SINGLE SLURM job that processes all mode
# displacement structures sequentially (both + and - displacements).
#
# Usage:
#   ./submit_mode_jobs.sh [structures_dir] [first_mode] [last_mode]
#
# Arguments:
#   structures_dir: Directory containing vasp_mode*.vasp files (default: structures)
#   first_mode: First mode number to submit (default: 1)
#   last_mode: Last mode number to submit (default: auto-detect)
#
# Example:
#   ./submit_mode_jobs.sh structures 1 60
#   ./submit_mode_jobs.sh  # Uses defaults
#
###############################################################################

# Default parameters
STRUCTURES_DIR="${1:-structures}"
FIRST_MODE="${2:-1}"
LAST_MODE="${3:-}"

# SLURM parameters - Based on run_1.sh
PARTITION="batch"               # Partition name
TIME="2-00:00:00"              # Time limit (2 days) - increase if needed for all modes
NTASKS=24                       # Number of MPI tasks
MEM_PER_CPU=3800               # Memory per CPU in MB
MAIL_USER="x.he@uliege.be"     # Your email

# Job script name
JOB_SCRIPT="job_all_modes.sh"

# Determine last mode if not specified
if [ -z "$LAST_MODE" ]; then
    # Count mode files (divide by 2 for +/- pairs, subtract 1 for mode0)
    MODE_COUNT=$(ls -1 "${STRUCTURES_DIR}"/vasp_mode*.vasp 2>/dev/null | wc -l)
    if [ $MODE_COUNT -eq 0 ]; then
        echo "ERROR: No structure files found in ${STRUCTURES_DIR}/"
        exit 1
    fi
    # Calculate last mode: (total files - 1 undisplaced) / 2 signs
    LAST_MODE=$(( (MODE_COUNT - 1) / 2 ))
fi

echo "================================================================================"
echo "CREATING SERIAL PHONON MODE CALCULATION JOB"
echo "================================================================================"
echo "Structures directory: ${STRUCTURES_DIR}"
echo "Mode range: ${FIRST_MODE} to ${LAST_MODE}"
echo "Partition: ${PARTITION}"
echo "Tasks per calculation: ${NTASKS}"
echo "Total calculations: $(( (LAST_MODE - FIRST_MODE + 1) * 2 ))"
echo "================================================================================"
echo ""

# Check if structures directory exists
if [ ! -d "${STRUCTURES_DIR}" ]; then
    echo "ERROR: Structures directory not found: ${STRUCTURES_DIR}"
    exit 1
fi

# Count how many calculations will be performed
TOTAL_CALCS=0
for MODE_NUM in $(seq $FIRST_MODE $LAST_MODE); do
    for SIGN in "+" "-"; do
        STRUCTURE_FILE="${STRUCTURES_DIR}/vasp_mode${MODE_NUM}${SIGN}.vasp"
        if [ -f "${STRUCTURE_FILE}" ]; then
            ((TOTAL_CALCS++))
        fi
    done
done

if [ $TOTAL_CALCS -eq 0 ]; then
    echo "ERROR: No valid structure files found in range"
    exit 1
fi

echo "Found ${TOTAL_CALCS} structure files to process"
echo ""

# Create the master job script
cat > "${JOB_SCRIPT}" << 'EOFMASTER'
#!/bin/bash
# Master job script for serial phonon mode calculations
#SBATCH --job-name=all_modes
#SBATCH --time=SLURM_TIME
#SBATCH --ntasks=SLURM_NTASKS
#SBATCH --mem-per-cpu=SLURM_MEM_PER_CPU
#SBATCH --partition=SLURM_PARTITION
#SBATCH --output=all_modes_slurm-%j.out
#SBATCH --mail-user=SLURM_MAIL_USER
#SBATCH --mail-type=END,FAIL

echo "================================================================================"
echo "SERIAL PHONON MODE CALCULATIONS"
echo "================================================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Started: $(date)"
echo "Node: $(hostname)"
echo "================================================================================"
echo ""

# Create scratch directory
mkdir -p $GLOBALSCRATCH/$SLURM_JOB_ID

# Module loading (uncomment and modify as needed)
#module --force purge
#module load releases/2022a
#module load imkl
#module load netCDF-Fortran/4.6.0-gompi-2022a
#module load Python/3.10.4-GCCcore-11.3.0-bare
#module load libxc/5.2.3-GCC-11.3.0
#module load FFTW.MPI/3.3.10-gompi-2022a
#module load OpenMPI/4.1.4-GCC-11.3.0
#module load OpenBLAS/0.3.20-GCC-11.3.0

# Set SIESTA command (modify for your system)
# export ASE_SIESTA_COMMAND="mpirun -n SLURM_NTASKS siesta"
# Or use default from environment (TB2Jflows handles this)

# Set thread affinity (optional, modify as needed)
# taskset -p 0xffffffee $$

# Counters
COMPLETED=0
FAILED=0
SKIPPED=0

# Loop over modes
for MODE_NUM in $(seq MODE_FIRST MODE_LAST); do
    for SIGN in "+" "-"; do
        # Define file paths
        STRUCTURE_FILE="STRUCTURES_DIR/vasp_mode${MODE_NUM}${SIGN}.vasp"
        OUTPUT_DIR="mode${MODE_NUM}${SIGN}"
        
        # Check if structure file exists
        if [ ! -f "${STRUCTURE_FILE}" ]; then
            echo "[$(date +%T)] ⚠ Warning: Structure file not found: ${STRUCTURE_FILE}"
            ((SKIPPED++))
            continue
        fi
        
        # Check if calculation already completed
        if [ -f "${OUTPUT_DIR}/calculation_summary.txt" ]; then
            echo "[$(date +%T)] ⊘ Skipping mode${MODE_NUM}${SIGN} (already completed)"
            ((SKIPPED++))
            continue
        fi
        
        # Create output directory
        mkdir -p "${OUTPUT_DIR}"
        
        # Run calculation
        echo "[$(date +%T)] ▶ Starting mode${MODE_NUM}${SIGN}..."
        python run_mode_calculation.py "${STRUCTURE_FILE}" "${OUTPUT_DIR}"
        
        # Check exit status
        if [ $? -eq 0 ]; then
            echo "[$(date +%T)] ✓ Completed mode${MODE_NUM}${SIGN}"
            ((COMPLETED++))
        else
            echo "[$(date +%T)] ✗ Failed mode${MODE_NUM}${SIGN}"
            ((FAILED++))
        fi
        
        echo ""
    done
done

echo "================================================================================"
echo "JOB COMPLETE"
echo "================================================================================"
echo "Finished: $(date)"
echo "Completed: ${COMPLETED}"
echo "Failed: ${FAILED}"
echo "Skipped: ${SKIPPED}"
echo "================================================================================"

# Exit with failure if any calculations failed
if [ $FAILED -gt 0 ]; then
    exit 1
else
    exit 0
fi
EOFMASTER

# Replace placeholders with actual values
sed -i.bak \
    -e "s|SLURM_TIME|${TIME}|g" \
    -e "s|SLURM_NTASKS|${NTASKS}|g" \
    -e "s|SLURM_MEM_PER_CPU|${MEM_PER_CPU}|g" \
    -e "s|SLURM_PARTITION|${PARTITION}|g" \
    -e "s|SLURM_MAIL_USER|${MAIL_USER}|g" \
    -e "s|MODE_FIRST|${FIRST_MODE}|g" \
    -e "s|MODE_LAST|${LAST_MODE}|g" \
    -e "s|STRUCTURES_DIR|${STRUCTURES_DIR}|g" \
    "${JOB_SCRIPT}"

# Remove backup file
rm -f "${JOB_SCRIPT}.bak"

# Make job script executable
chmod +x "${JOB_SCRIPT}"

echo "Created job script: ${JOB_SCRIPT}"
echo ""
echo "To submit the job, run:"
echo "  sbatch ${JOB_SCRIPT}"
echo ""
echo "Or submit now? (y/n)"
read -r SUBMIT_NOW

if [ "$SUBMIT_NOW" = "y" ] || [ "$SUBMIT_NOW" = "Y" ]; then
    JOB_ID=$(sbatch --parsable "${JOB_SCRIPT}")
    
    if [ $? -eq 0 ]; then
        echo ""
        echo "✓ Job submitted successfully"
        echo "  Job ID: ${JOB_ID}"
        echo "  Output: all_modes_slurm-${JOB_ID}.out"
        echo ""
        echo "To check job status: squeue -j ${JOB_ID}"
        echo "To cancel job: scancel ${JOB_ID}"
        echo "To monitor output: tail -f all_modes_slurm-${JOB_ID}.out"
    else
        echo ""
        echo "✗ Failed to submit job"
        exit 1
    fi
else
    echo ""
    echo "Job script created but not submitted."
    echo "Submit manually with: sbatch ${JOB_SCRIPT}"
fi

echo "================================================================================"
