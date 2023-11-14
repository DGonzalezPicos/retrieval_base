#!/bin/bash

# Set job requirements
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH -t 03:00:00
#SBATCH -p genoa
#SBATCH --ntasks=128
#SBATCH --mem=336G

#SBATCH --job-name=fiducial_first
#SBATCH --mail-type=ALL
#SBATCH --mail-user=picos@strw.leidenuniv.nl

# Loading modules
module load 2022
module load Python/3.10.4-GCCcore-11.3.0
module load OpenBLAS/0.3.20-GCC-11.3.0
module load OpenMPI/4.1.4-GCC-11.3.0
module load libarchive/3.6.1-GCCcore-11.3.0

# Activate virtual environment
# source $HOME/retrieval_venv/bin/activate
conda activate retrieval

# Export environment variables
export LD_LIBRARY_PATH=$HOME/retrieval_venv/MultiNest/lib:$LD_LIBRARY_PATH
export pRT_input_data_path=$HOME/pRT/input_data

echo "Number of tasks $SLURM_NTASKS"
echo "Starting Python script"

# Replace the config file and run pre-processing
sed -i 's/import config as conf/import config_fiducial as conf/g' retrieval_script.py
python retrieval_script.py -p

# Run the retrieval and evaluation
mpiexec -np 128 --bind-to core python retrieval_script.py -r
mpiexec -np 128 --bind-to core python retrieval_script.py -e

# Revert to original config file
sed -i 's/import config_fiducial as conf/import config as conf/g' retrieval_script.py

echo "Done"
