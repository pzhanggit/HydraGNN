#!/bin/bash
#BSUB -env "all"
#BSUB -P LRN026
#BSUB -J HydraGNN
#BSUB -W 8:00
#BSUB -nnodes 1
#BSUB -q batch-hm
#BSUB -o vib-%J.out
#BSUB -e vib-%J.out

[ -z $JOBID ] && JOBID=$LSB_JOBID
[ -z $JOBSIZE ] && JOBSIZE=$(((LSB_DJOB_NUMPROC-1)/42))

runit () {
    CMD="$@"
    echo "CMD: $CMD"
    "$@"
}

module purge
ml DefApps
ml gcc
module unload darshan-runtime
module use -a /gpfs/alpine/world-shared/csc143/jyc/summit/sw/modulefiles
ml anaconda3/2022.10
ml adios2/devel
export LD_LIBRARY_PATH=/lib64:/usr/lib64:$LD_LIBRARY_PATH
export PYTHONPATH=$PWD:$PYTHONPATH

NR=1 #6
NP=$((JOBSIZE*NR))
export HYDRAGNN_BACKEND=nccl

#runit jsrun -n$NP -a1 -c$((42/NR)) -g1 -r$NR -brs python -u examples/vibrational_spectroscopy/vibrational_spectroscopy_test.py --testplotskip=100 --logname="qm8-SMILES-AD-WLoss-SMLlr-XSNN-NoMend-NoSMILES-schnet" --normalizey --loadsplit --inputfile=vibrational_spectroscopy_qm8_XS_pca_noSMILES_schnet.json &

#runit jsrun -n$NP -a1 -c$((42/NR)) -g1 -r$NR -brs python -u examples/vibrational_spectroscopy/vibrational_spectroscopy_test.py --testplotskip=100 --logname="qm8-SMILES-AD-WLoss-SMLlr-XSNN-NoMend-schnet" --normalizey --loadsplit --inputfile=vibrational_spectroscopy_qm8_XS_pca_schnet.json &

#runit jsrun -n$NP -a1 -c$((42/NR)) -g1 -r$NR -brs python -u examples/vibrational_spectroscopy/vibrational_spectroscopy_test.py --testplotskip=100 --logname="qm8-SMILES-AD-WLoss-SMLlr-XSNN-Mend-schnet" --normalizey --loadsplit --inputfile=vibrational_spectroscopy_qm8_XS_pca_schnet.json  --mendeleev & 

#runit jsrun -n$NP -a1 -c$((42/NR)) -g1 -r$NR -brs python -u examples/vibrational_spectroscopy/vibrational_spectroscopy_test.py --testplotskip=100 --logname="qm8-SMILES-AD-WLoss-SMLlr-XSNN-Mend-pca72-schnet" --normalizey --loadsplit --pca=72 --inputfile=vibrational_spectroscopy_qm8_XS_pca_schnet.json  --mendeleev & 

#runit jsrun -n$NP -a1 -c$((42/NR)) -g1 -r$NR -brs python -u examples/vibrational_spectroscopy/vibrational_spectroscopy_test.py --testplotskip=100 --logname="qm8-SMILES-AD-WLoss-SMLlr-XSNN-Mend-ae42-schnet" --normalizey --loadsplit --mendeleev --pca=-42 --inputfile=vibrational_spectroscopy_qm8_XS_pca_schnet.json &

#runit jsrun -n$NP -a1 -c$((42/NR)) -g1 -r$NR -brs python -u examples/vibrational_spectroscopy/vibrational_spectroscopy_test.py --testplotskip=100 --logname="qm8-SMILES-AD-WLoss-SMLlr-XSNN-Mend-ae42-schnet-NoBN" --normalizey --loadsplit --mendeleev --pca=-42 --inputfile=vibrational_spectroscopy_qm8_XS_pca_schnet.json &

runit jsrun -n$NP -a1 -c$((42/NR)) -g1 -r$NR -brs python -u examples/vibrational_spectroscopy/vibrational_spectroscopy_test.py --testplotskip=100 --logname="qm8-SMILES-AD-WLoss-SMLlr-XSNN-Mend-ae42-schnet-NoBN-400" --normalizey --loadsplit --mendeleev --pca=-42 --inputfile=vibrational_spectroscopy_qm8_XS_pca_schnet.json &
wait
