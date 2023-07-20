# bash

for d in 64 32 16 8; do
    qsub qsub.sh python experiment/dnn_norm_experiment.py\
        --num_results 1000 \
        --d $d
done

for signal in 1.0 2.0 3.0 4.0; do
    qsub qsub.sh python experiment/dnn_norm_experiment.py\
        --num_results 1000 \
        --signal $signal
done

for num_samples in 400 300 200 100; do
    qsub qsub.sh python experiment/sfs_norm_experiment.py\
        --num_results 1000 \
        --num_samples $num_samples
done

for signal in 0.1 0.2 0.3 0.4; do
    qsub qsub.sh python experiment/sfs_norm_experiment.py\
        --num_results 1000 \
        --signal $signal
done

for num_samples in 400 300 200 100; do
    qsub qsub.sh python experiment/sfs_chi_experiment.py\
        --num_results 1000 \
        --num_samples $num_samples
done

for signal in 0.1 0.2 0.3 0.4; do
    qsub qsub.sh python experiment/sfs_chi_experiment.py\
        --num_results 1000 \
        --signal $signal
done
