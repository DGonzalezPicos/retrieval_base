#bin/bash
# pass two arguments to the script: the input run and output run
input_run=$1
output_run=$2
# create the output directory
mkdir -p retrieval_outputs/$output_run
# create two subdirectories in the output run: test_data and test_plots
mkdir -p retrieval_outputs/$output_run/test_data
mkdir -p retrieval_outputs/$output_run/test_plots
# copy the file pRT_atm_NIRSpec.pkl from input_run/test_data to output_run/test_data
cp -r retrieval_outputs/$input_run/test_data/pRT_atm_NIRSpec.pkl retrieval_outputs/$output_run/test_data
# print the message "Done copying the atmospheric file"
echo "Done copying the atmospheric file"
