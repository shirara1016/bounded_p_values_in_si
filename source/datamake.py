import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, ".."))

import tensorflow as tf
import numpy as np
import source.models as models
from dnn_norm import DnnSelectiveInferenceNorm
from tqdm import tqdm
import pickle


num_iter = 1050
rng = np.random.default_rng(seed=0)


def make_dataset(d, signal):
    a = d // 3

    if d <= 16:
        buffer_size = int(30 * num_iter)
    elif d == 32:
        buffer_size = int(10 * num_iter)
    else:
        buffer_size = int(3 * num_iter)

    buffer = []
    for _ in tqdm(range(buffer_size)):
        X = rng.normal(0, 1, (d, d, 1))
        abnormal_x = np.random.randint(0, d - a)
        abnormal_y = np.random.randint(0, d - a)
        X[abnormal_x : abnormal_x + a, abnormal_y : abnormal_y + a, 0] = rng.normal(
            signal, 1, (a, a)
        )
        buffer.append(X)
    buffer = np.array(buffer)

    shape = (d, d, 1)

    model = models.simple_model_classification(shape)
    model.load_weights(f"models/cam_d{d}.h5")
    layers = model.layers
    cam = models.CAM(layers[-1], shape)([layers[-3].output, layers[-1].output])
    model_with_cam = tf.keras.Model(inputs=model.inputs, outputs=cam)

    si = DnnSelectiveInferenceNorm(model_with_cam, d)

    buffer_saliency = (
        si.cam_model.forward(buffer).numpy().reshape((buffer_size, -1)) >= 0
    )
    densed_map = np.mean(buffer_saliency, axis=1)
    index = (densed_map > 0) * (densed_map < 1)

    buffer = buffer[index]
    assert len(buffer) > num_iter

    dataset = buffer[:num_iter]
    dataset = [dataset[i] for i in range(num_iter)]
    with open(f"dnn_dataset/seed{0}_d{d}_delta{signal}.pkl", "wb") as f:
        pickle.dump(dataset, f)

    dataset = np.array(dataset)
    print(dataset.shape)

    return None


for d in [8, 16, 32, 64]:
    make_dataset(d=d, signal=0.0)

for signal in [1.0, 2.0, 3.0, 4.0]:
    make_dataset(d=16, signal=signal)
