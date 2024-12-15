# Installing petitRADTRANS using the 2023 modules

# from $HOME/retrieval_base (2024-12-13)
source modules23.sh
pip install numpy==1.23.5
pip install petitRADTRANS==2.6.7
pip install mpi4py
pip install corner==2.2.2 matplotlib==3.9.2 PyAstronomy==0.18.1 pymultinest==2.11
# install local packages: retrieval_base, retsupjup, broadpy with `pip install -e .`
pip install spectres==2.2.0
pip install wget
pip install line_profiler xarray
# test
python -c "from petitRADTRANS import Radtrans"