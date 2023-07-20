#!/bin/csh
#$ -cwd
#$ -V -S /bin/bash
#$ -N pp_based_si
#$ -q cpu-e.q@*
#$ -pe smp 50
#$ -o path_for_stdoutfile
#$ -e path_for_errorfile
# export OMP_NUM_THREADS = 1

$@
