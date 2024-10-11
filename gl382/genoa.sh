#!/bin/bash
# Set job requirements
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH -t 02:59:30
#SBATCH -p genoa
#SBATCH --ntasks=134
#SBATCH --mem=336G


#SBATCH --job-name=fc2
#SBATCH --mail-type=ALL
#SBATCH --mail-user=picos@strw.leidenuniv.nl

# Loading modules
module load 2022
module load Python/3.10.4-GCCcore-11.3.0
module load OpenBLAS/0.3.20-GCC-11.3.0
module load OpenMPI/4.1.4-GCC-11.3.0
module load libarchive/3.6.1-GCCcore-11.3.0


# Export environment variables
export OMPI_MCA_pml=ucx
export LD_LIBRARY_PATH=$HOME/MultiNest/lib:$LD_LIBRARY_PATH
export pRT_input_data_path=/projects/0/prjs1096/pRT/input_data

echo "Number of tasks $SLURM_NTASKS"
echo "Starting Python script"

# define variable target
target=gl382

mpiexec -np $SLURM_NTASKS --bind-to core python retrieval_script.py -r -t $target
echo "Done"
