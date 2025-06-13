#!/bin/bash
#SBATCH -A LRN070
#SBATCH -J HydraGNN
#SBATCH -o test-%j-bf16.out
#SBATCH -e test-%j-bf16.out
#SBATCH -t 00:10:00
#SBATCH -p batch 
##SBATCH -q debug
#SBATCH -N 1 
##SBATCH -S 1

 
# Load conda environemnt
#source /lustre/orion/lrn070/world-shared/mlupopa/module-to-load-frontier-rocm624.sh
#source /lustre/orion/lrn070/world-shared/mlupopa/max_conda_envs_frontier/bin/activate
#conda activate hydragnn_rocm624
 
#export python path to use ADIOS2 v.2.9.2
#export PYTHONPATH=/lustre/orion/lrn070/world-shared/mlupopa/ADIOS_ROCm624/adios2-install/lib/python3.11/site-packages/:$PYTHONPATH

# Load conda environment
source /lustre/orion/lrn070/world-shared/mlupopa/module-to-load-frontier-rocm631.sh
source /lustre/orion/lrn070/world-shared/mlupopa/max_conda_envs_frontier/bin/activate
conda activate hydragnn_rocm631


#export python path to use ADIOS2 v.2.10.2
export PYTHONPATH=/lustre/orion/lrn070/world-shared/mlupopa/ADIOS_ROCm631/adios2-install/lib/python3.11/site-packages/:$PYTHONPATH

export LD_PRELOAD=/lib64/libgcc_s.so.1:/usr/lib64/libstdc++.so.6

 

which python
python -c "import numpy; print(numpy.__version__)"


echo $LD_LIBRARY_PATH  | tr ':' '\n'

export MPICH_ENV_DISPLAY=0
export MPICH_VERSION_DISPLAY=0
export MIOPEN_DISABLE_CACHE=1
export NCCL_PROTO=Simple

export OMP_NUM_THREADS=7
export HYDRAGNN_NUM_WORKERS=0
export HYDRAGNN_USE_VARIABLE_GRAPH_SIZE=1
export HYDRAGNN_AGGR_BACKEND=mpi

#export HYDRAGNN_TRACE_LEVEL=1
#export HYDRAGNN_MAX_NUM_BATCH=5

export NCCL_P2P_LEVEL=NVL
export NCCL_P2P_DISABLE=1


## Checking
env | grep ROCM
env | grep ^MI
env | grep ^MPICH
env | grep ^HYDRA

# Beginning of job: enable data collection
ml use /autofs/nccs-svm1_sw/crusher/amdsw/modules
ml omnistat-wrapper
export OMNISTAT_CONFIG=${OMNISTAT_DIR}/omnistat/config/omnistat.ornl.external
${OMNISTAT_WRAPPER} usermode --start --interval 0.01 #5
# Launch your application(s) as normal
srun -N$SLURM_JOB_NUM_NODES -n$((SLURM_JOB_NUM_NODES*8)) -c7 --gpus-per-task=1 --gpu-bind=closest python -u ./examples/qm9/qm9.py --bf16  --log="qm9-bf16-omnistat" #--log="qm9-bf16"
# End of job: stop data collection
${OMNISTAT_WRAPPER} usermode --stop
