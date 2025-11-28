# SLURM Job Submission Scripts

This directory contains scripts for submitting phonon mode displacement calculations to a SLURM cluster.

## Files

- **`run_mode_calculation.py`**: Python script that runs a single DFT calculation for one mode displacement
- **`submit_mode_jobs.sh`**: Bash script that submits all mode calculations as separate SLURM jobs

## Setup

### 1. Configure SLURM Parameters

Edit `submit_mode_jobs.sh` to set your cluster-specific parameters:

```bash
PARTITION="batch"               # Partition to use
TIME="2-00:00:00"              # Time limit per job (2 days)
NTASKS=24                       # MPI tasks per job
MEM_PER_CPU=3800               # Memory per CPU (MB)
MAIL_USER="jiahui.jia@uliege.be"  # Email for job notifications
```

**Note:** These parameters are already configured to match `run_1.sh`.

### 2. Configure SIESTA Settings

Edit `run_mode_calculation.py` to configure your SIESTA+TB2J calculation:

```python
# Rare earth element (auto-detected from structure)
Re = "Tm"  # or "Sm", "Dy", etc.

# DFT+U parameters
Udict = {"Fe": dict(n=3, l=2, U=4.5, J=0, rc=2.11)}
if Re != "La":
    Udict[Re] = dict(n=4, l=3, U=Uf[Re], J=0.0, rc=2.5)

# K-point mesh
kmesh = [7, 5, 7]

# Spin treatment
spin = "spin-orbit"  # or "non-collinear"
```

### 3. Load Required Modules

In the generated job scripts, uncomment and modify the module loading section as needed:

```bash
#module --force purge
#module load releases/2022a
#module load Python/3.10.4-GCCcore-11.3.0-bare
# ... etc
```

These are already configured to match your cluster setup from `run_1.sh`.

### 4. Set SIESTA Command

In `submit_mode_jobs.sh`, set the SIESTA execution command:

```bash
export ASE_SIESTA_COMMAND="mpirun -n ${NTASKS} siesta"
```

Or use the default from your environment. The script uses TB2Jflows which handles SIESTA execution.

## Usage

### Generate Mode Structures

First, generate the phonon mode displacement structures:

```bash
python example_mode_summary_and_thermal.py path/to/phonopy_params.yaml
```

This creates structures in `structures/`:
- `vasp_mode0.vasp` (undisplaced)
- `vasp_mode1+.vasp`, `vasp_mode1-.vasp`
- `vasp_mode2+.vasp`, `vasp_mode2-.vasp`
- ... etc

### Submit All Jobs

Submit all mode calculations:

```bash
./submit_mode_jobs.sh
```

This will:
- Create individual job scripts for each mode displacement
- Submit each as a separate SLURM job
- Create output directories `mode1+/`, `mode1-/`, etc.

### Submit Specific Mode Range

Submit only modes 1-10:

```bash
./submit_mode_jobs.sh structures 1 10
```

### Submit from Different Directory

```bash
./submit_mode_jobs.sh path/to/structures 1 60
```

## Output Structure

Each calculation creates a directory with:

```
mode1+/
├── siesta.fdf                # SIESTA input file
├── siesta.out                # SIESTA output
├── siesta.XV                 # Atomic positions and velocities
├── siesta.DM                 # Density matrix
├── TB2J/                     # TB2J results directory
├── calculation_summary.txt   # Summary of results
├── slurm-12345.out          # SLURM stdout
└── slurm-12345.err          # SLURM stderr
```

## Monitoring Jobs

Check job status:
```bash
squeue -u $USER
```

Cancel all your jobs:
```bash
scancel -u $USER
```

Cancel specific job:
```bash
scancel <job_id>
```

View job output:
```bash
tail -f mode1+/slurm-*.out
```

## Resubmitting Failed Jobs

The script automatically skips completed calculations (those with `calculation_summary.txt`).

To resubmit failed jobs, delete the summary file and rerun:

```bash
rm mode5+/calculation_summary.txt
./submit_mode_jobs.sh structures 5 5  # Resubmit only mode 5+
```

## Calculation Details

### SIESTA+TB2J Setup

The script uses `TB2Jflows.auto_siesta_TB2J` which:
- Runs SIESTA with spin-orbit coupling or non-collinear magnetism
- Applies DFT+U for Fe 3d and rare earth 4f electrons
- Calculates magnetic exchange interactions with TB2J
- Supports various rare earth elements (Tm, Sm, Dy, etc.)

### Magnetic Configuration

Default: C-AFM (C-type antiferromagnetic)
- Fe atoms: [4, 4, -4, -4] (in μ_B)
- Rare earth: [m, -m, -m, m] where m depends on the element
- Oxygen: non-magnetic [0, 0, ..., 0]

### Customization for Different Materials

To adapt for different materials, modify `run_mode_calculation.py`:

1. **Change magnetic moments** (line ~60):
   ```python
   magmoms = [your_custom_moments]
   ```

2. **Change DFT+U parameters** (line ~70):
   ```python
   Udict = {"Element": dict(n=3, l=2, U=4.5, J=0, rc=2.11)}
   ```

3. **Change k-point mesh** (line ~95):
   ```python
   kmesh = [7, 5, 7]  # Adjust for your cell
   ```

## Notes

- The script creates one SLURM job per displacement structure
- Job names are set to `mode1+`, `mode1-`, etc. for easy identification
- Failed jobs can be identified by missing `calculation_summary.txt` files
- Adjust memory and time limits based on your system size

## Example Workflow

```bash
# 1. Generate structures
python example_mode_summary_and_thermal.py phonopy_params.yaml

# 2. Test one calculation locally (optional)
python run_mode_calculation.py structures/vasp_mode1+.vasp test_mode1+

# 3. Submit all jobs
./submit_mode_jobs.sh

# 4. Monitor progress
watch -n 60 'squeue -u $USER'

# 5. Check for failures
for dir in mode*/; do
    if [ ! -f "$dir/calculation_summary.txt" ]; then
        echo "Failed or incomplete: $dir"
    fi
done
```
