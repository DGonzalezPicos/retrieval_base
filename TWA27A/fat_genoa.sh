#!/bin/bash

# Set job requirements
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH -t 07:00:00
#SBATCH -p fat_genoa
#SBATCH --ntasks=192

#SBATCH --job-name=lbl15_G1_1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=picos@strw.leidenuniv.nl

# Loading modules
source $HOME/retrieval_base/modules23.sh


# Export environment variables
export OMPI_MCA_pml=ucx
export LD_LIBRARY_PATH=$HOME/MultiNest/lib:$LD_LIBRARY_PATH
export pRT_input_data_path=/projects/0/prjs1096/pRT/input_data

echo "Number of tasks $SLURM_NTASKS"
echo "Starting Python script"

mpiexec -np $SLURM_NTASKS --bind-to core python retrieval_script_jwst.py -r

echo "Done"
