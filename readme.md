# Bounded P-values in Parametric Programming-based Selective Inference
This pacakge is the implementation of the paper "Bounded P-values in Parametric Programming-based Selective Inference" for experiments.

## Installation & Requirements
This pacakage has the following dependencies:
- Python (version 3.10 or higher, we use 3.10.11)
- sicore (we use version 1.0.0)
- tensorflow (we use version 2.11.1)
- tqdm

Please install these dependencies by pip.
```
pip install sicore
pip install tensorflow
pip install tqdm
```

## Reproducibility

Since we have already got the results in advance, you can reproduce the figures by running following code. The results will be saved in "/image" folder.
```
sh visualize.sh
```

To reproduce the results, please see the following instructions after installation step.
The results will be saved in "./results" folder as pickle file.

For the selective $z$-test in SFS.
```
sh ex_sfs_norm.sh
```

For the selective $\chi$-test in SFS.
```
sh ex_sfs_chi.sh
```

For the selective $z$-test in DNN.
```
sh ex_dnn_norm.sh
```

For visualization of the reproduced results.
```
sh visualize.sh
```
