# bash

for d in 64 32 16 8; do
    python experiment/dnn_norm_experiment.py\
        --num_results 1000 \
        --d $d
done

for signal in 1.0 2.0 3.0 4.0; do
    python experiment/dnn_norm_experiment.py\
        --num_results 1000 \
        --signal $signal
done
