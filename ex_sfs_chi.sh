# bash

for num_samples in 400 300 200 100; do
    python experiment/sfs_chi_experiment.py\
        --num_results 1000 \
        --num_samples $num_samples
done

for signal in 0.1 0.2 0.3 0.4; do
    python experiment/sfs_chi_experiment.py\
        --num_results 1000 \
        --signal $signal
done
