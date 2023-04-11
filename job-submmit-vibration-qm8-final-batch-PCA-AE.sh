#!/bin/bash
#BSUB -env "all"
#BSUB -P MAT250
#BSUB -J HydraGNN
#BSUB -W 2:00
#BSUB -nnodes 1
#BSUB -q batch
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

module use -a /gpfs/alpine/world-shared/csc143/jyc/summit/sw/modulefiles
ml envs/py38
ml boost/1.78.0
ml rdkit/devel
ml adios2/devel
ml papi

export LD_PRELOAD=/sw/summit/gcc/9.1.0-alpha+20190716/lib64/libstdc++.so:/sw/summit/gcc/9.1.0-alpha+20190716/lib64/libgomp.so 
export PYTHONPATH=$PWD:$PYTHONPATH

NR=1 #6
NP=$((JOBSIZE*NR))
export HYDRAGNN_BACKEND=nccl



#########################
#runit jsrun -n$NP -a1 -c$((42/NR)) -g1 -r$NR -brs python -u examples/vibrational_spectroscopy/vibrational_spectroscopy_test.py --testplotskip=100 --logname="qm8-SMILES-AD-WLoss-SMLlr-XSNN-Mend-NoSMILES" --normalizey --loadsplit --inputfile=vibrational_spectroscopy_qm8_XS_pca_noSMILES.json  --mendeleev & 

#runit jsrun -n$NP -a1 -c$((42/NR)) -g1 -r$NR -brs python -u examples/vibrational_spectroscopy/vibrational_spectroscopy_test.py --testplotskip=100 --logname="qm8-SMILES-AD-WLoss-SMLlr-XSNN-NoMend-NoSMILES" --normalizey --loadsplit --inputfile=vibrational_spectroscopy_qm8_XS_pca_noSMILES.json &
#wait
#########################
#runit jsrun -n$NP -a1 -c$((42/NR)) -g1 -r$NR -brs python -u examples/vibrational_spectroscopy/vibrational_spectroscopy_test.py --testplotskip=100 --logname="qm8-SMILES-AD-WLoss-SMLlr-XSNN-Mend" --normalizey --loadsplit --inputfile=vibrational_spectroscopy_qm8_XS_pca.json  --mendeleev & 

#runit jsrun -n$NP -a1 -c$((42/NR)) -g1 -r$NR -brs python -u examples/vibrational_spectroscopy/vibrational_spectroscopy_test.py --testplotskip=100 --logname="qm8-SMILES-AD-WLoss-SMLlr-XSNN-NoMend" --normalizey --loadsplit --inputfile=vibrational_spectroscopy_qm8_XS_pca.json &
#wait
#########################
runit jsrun -n$NP -a1 -c$((42/NR)) -g1 -r$NR -brs python -u examples/vibrational_spectroscopy/vibrational_spectroscopy_test.py --testplotskip=100 --logname="qm8-SMILES-AD-WLoss-SMLlr-XSNN-NoMend-pca72" --normalizey --loadsplit --pca=72 --inputfile=vibrational_spectroscopy_qm8_XS_pca.json 

runit jsrun -n$NP -a1 -c$((42/NR)) -g1 -r$NR -brs python -u examples/vibrational_spectroscopy/vibrational_spectroscopy_test.py --testplotskip=100 --logname="qm8-SMILES-AD-WLoss-SMLlr-XSNN-Mend-pca72" --normalizey --loadsplit --pca=72 --inputfile=vibrational_spectroscopy_qm8_XS_pca.json  --mendeleev  

#wait
#########################
#runit jsrun -n$NP -a1 -c$((42/NR)) -g1 -r$NR -brs python -u examples/vibrational_spectroscopy/vibrational_spectroscopy_test.py --testplotskip=100 --logname="qm8-SMILES-AD-WLoss-SMLlr-XSNN-NoMend-ae30" --normalizey --loadsplit --pca=-30 --inputfile=vibrational_spectroscopy_qm8_XS_pca.json &

#runit jsrun -n$NP -a1 -c$((42/NR)) -g1 -r$NR -brs python -u examples/vibrational_spectroscopy/vibrational_spectroscopy_test.py --testplotskip=100 --logname="qm8-SMILES-AD-WLoss-SMLlr-XSNN-NoMend-ae36" --normalizey --loadsplit --pca=-36 --inputfile=vibrational_spectroscopy_qm8_XS_pca.json &
#wait

#runit jsrun -n$NP -a1 -c$((42/NR)) -g1 -r$NR -brs python -u examples/vibrational_spectroscopy/vibrational_spectroscopy_test.py --testplotskip=100 --logname="qm8-SMILES-AD-WLoss-SMLlr-XSNN-NoMend-ae42" --normalizey --loadsplit --pca=-42 --inputfile=vibrational_spectroscopy_qm8_XS_pca.json &
#########################
#runit jsrun -n$NP -a1 -c$((42/NR)) -g1 -r$NR -brs python -u examples/vibrational_spectroscopy/vibrational_spectroscopy_test.py --testplotskip=100 --logname="qm8-SMILES-AD-WLoss-SMLlr-XSNN-Mend-ae36" --normalizey --loadsplit --mendeleev --pca=-36 --inputfile=vibrational_spectroscopy_qm8_XS_pca.json &
runit jsrun -n$NP -a1 -c$((42/NR)) -g1 -r$NR -brs python -u examples/vibrational_spectroscopy/vibrational_spectroscopy_test.py --testplotskip=100 --logname="qm8-SMILES-AD-WLoss-SMLlr-XSNN-Mend-ae42" --normalizey --loadsplit --mendeleev --pca=-42 --inputfile=vibrational_spectroscopy_qm8_XS_pca.json 
#########################
