#!/bin/bash
#SBATCH -A CPH161
#SBATCH -J HydraGNN
#SBATCH -o job-%j.out
#SBATCH -e job-%j.out
#SBATCH -t 00:10:00
#SBATCH -p batch
#SBATCH -q debug
##SBATCH -N 64  #16
#SBATCH -N 2 #32
#SBATCH -S 1
#SBATCH --exclude=frontier01252,frontier00444

module reset
module load PrgEnv-gnu
module load rocm/5.7.1
module load cmake
module load craype-accel-amd-gfx90a
module load amd-mixed/5.7.1
module load cray-mpich/8.1.26
module load miniforge3/23.11.0
module unload darshan-runtime

source activate /lustre/orion/world-shared/cph161/jyc/frontier/sw/envs/hydragnn-py39-rocm571-amd

## Use ADM build
PYTORCH_DIR=/autofs/nccs-svm1_sw/crusher/amdsw/karldev/pytorch-2.2.2-rocm5.7.1
PYG_DIR=/autofs/nccs-svm1_sw/crusher/amdsw/karldev/pyg-rocm5.7.1
export PYTHONPATH=$PYG_DIR:$PYTORCH_DIR:$PYTHONPATH

module use -a /lustre/orion/world-shared/cph161/jyc/frontier/sw/modulefiles
module load adios2/2.9.2-mpich-8.1.26

module load aws-ofi-rccl/devel-rocm5.7.1
echo $LD_LIBRARY_PATH  | tr ':' '\n'

export MPICH_ENV_DISPLAY=0
export MPICH_VERSION_DISPLAY=0
export MIOPEN_DISABLE_CACHE=1
export NCCL_PROTO=Simple

export OMP_NUM_THREADS=7
export HYDRAGNN_NUM_WORKERS=0
export HYDRAGNN_USE_VARIABLE_GRAPH_SIZE=1
export HYDRAGNN_AGGR_BACKEND=mpi

export NCCL_P2P_LEVEL=NVL
export NCCL_P2P_DISABLE=1


## Checking
env | grep ROCM
env | grep ^MI
env | grep ^MPICH
env | grep ^HYDRA

#export PYTHONPATH=/dir/to/HydraGNN:$PYTHONPATH
export PYTHONPATH=/lustre/orion/cph161/proj-shared/zhangp/HydraGNN_EL:$PYTHONPATH

#srun -N$SLURM_JOB_NUM_NODES -n$((SLURM_JOB_NUM_NODES*8)) -c7 --gpus-per-task=1 --gpu-bind=closest python -u ./examples/ensemble_learning/inference_alloy.py --models_dir_folder=examples/ensemble_learning/alloy_binary_rmsd_1,examples/ensemble_learning/alloy_binary_rmsd_2  --dataname=alloy_binary_rmsd --log="EL_rmsd3"
#srun -N$SLURM_JOB_NUM_NODES -n$((SLURM_JOB_NUM_NODES*8)) -c7 --gpus-per-task=1 --gpu-bind=closest python -u ./examples/ensemble_learning/inference_alloy.py --models_dir_folder=examples/ensemble_learning/alloy_binary_energy   --dataname=alloy_binary_energy   --log="EL_energy3"

srun -N$SLURM_JOB_NUM_NODES -n$((SLURM_JOB_NUM_NODES*8)) -c7 --gpus-per-task=1 --gpu-bind=closest python -u ./examples/ensemble_learning/inference_alloy.py --models_dir_folder=examples/ensemble_learning/alloy_ternary_rmsd  --dataname=alloy_ternary_rmsd --log="EL_ternary_rmsd3"
srun -N$SLURM_JOB_NUM_NODES -n$((SLURM_JOB_NUM_NODES*8)) -c7 --gpus-per-task=1 --gpu-bind=closest python -u ./examples/ensemble_learning/inference_alloy.py --models_dir_folder=examples/ensemble_learning/alloy_ternary_energy   --dataname=alloy_ternary_energy   --log="EL_ternary_energy3"

#srun -N$SLURM_JOB_NUM_NODES -n$((SLURM_JOB_NUM_NODES*8)) -c7 --gpus-per-task=1 --gpu-bind=closest python -u ./examples/ensemble_learning/inference_alloy.py --models_dir_folder=examples/ensemble_learning/alloy_ternary_rmsd  --dataname=alloy_ternary_rmsd --log="EL_ternary_rmsd"
#srun -N$SLURM_JOB_NUM_NODES -n$((SLURM_JOB_NUM_NODES*8)) -c7 --gpus-per-task=1 --gpu-bind=closest python -u ./examples/ensemble_learning/inference_alloy.py --models_dir_folder=examples/ensemble_learning/alloy_ternary_energy   --dataname=alloy_ternary_energy   --log="EL_ternary_energy"
#srun -N$SLURM_JOB_NUM_NODES -n$((SLURM_JOB_NUM_NODES*8)) -c7 --gpus-per-task=1 --gpu-bind=closest python -u ./examples/ensemble_learning/inference_alloy.py --models_dir_folder=examples/ensemble_learning/alloy_ternary_lattice --dataname=alloy_ternary_lattice --log="EL_ternary_lattice"

#srun -N$SLURM_JOB_NUM_NODES -n$((SLURM_JOB_NUM_NODES*8)) -c7 --gpus-per-task=1 --gpu-bind=closest python -u ./examples/ensemble_learning/inference_alloy.py --models_dir_folder=examples/ensemble_learning/alloy_binary_rmsd_1,examples/ensemble_learning/alloy_binary_rmsd_2  --dataname=alloy_binary_rmsd --log="EL_rmsd2"
#srun -N$SLURM_JOB_NUM_NODES -n$((SLURM_JOB_NUM_NODES*8)) -c7 --gpus-per-task=1 --gpu-bind=closest python -u ./examples/ensemble_learning/inference_alloy.py --models_dir_folder=examples/ensemble_learning/alloy_binary_energy   --dataname=alloy_binary_energy   --log="EL_energy2"
#srun -N$SLURM_JOB_NUM_NODES -n$((SLURM_JOB_NUM_NODES*8)) -c7 --gpus-per-task=1 --gpu-bind=closest python -u ./examples/ensemble_learning/inference_alloy.py --models_dir_folder=examples/ensemble_learning/alloy_binary_lattice --dataname=alloy_binary_lattice --log="EL_lattice2"


#srun -N$SLURM_JOB_NUM_NODES -n$((SLURM_JOB_NUM_NODES*8)) -c7 --gpus-per-task=1 --gpu-bind=closest python -u ./examples/ensemble_learning/inference_alloy.py --models_dir_folder=examples/ensemble_learning/alloy_binary_lattice --dataname=alloy_binary_lattice --log="EL_lattice"
#srun -N$SLURM_JOB_NUM_NODES -n$((SLURM_JOB_NUM_NODES*8)) -c7 --gpus-per-task=1 --gpu-bind=closest python -u ./examples/ensemble_learning/inference_alloy.py --models_dir_folder=examples/ensemble_learning/alloy_binary_rmsd_1,examples/ensemble_learning/alloy_binary_rmsd_2  --dataname=alloy_binary_rmsd --log="EL_rmsd"
#srun -N$SLURM_JOB_NUM_NODES -n$((SLURM_JOB_NUM_NODES*8)) -c7 --gpus-per-task=1 --gpu-bind=closest python -u ./examples/ensemble_learning/inference_alloy.py --models_dir_folder=examples/ensemble_learning/alloy_binary_energy --log="EL_energy"