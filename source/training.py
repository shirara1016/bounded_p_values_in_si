import tensorflow as tf
import numpy as np

import source.models as models


def make_training_dataset(rng, size, shape):
    X_train, y_train = [], []

    for _ in range(int(size * 0.4)):
        X = rng.normal(0, 1, shape)
        X_train.append(X)
        y_train.append(0)

    for _ in range(int(size * 0.6)):
        a = shape[0] // 4

        X = rng.normal(0, 1, shape)
        abnormal_x = rng.integers(0, shape[0] - a)
        abnormal_y = rng.integers(0, shape[1] - a)
        X[abnormal_x : abnormal_x + a, abnormal_y : abnormal_y + a, 0] = rng.normal(
            2, 1, (a, a)
        )

        X_train.append(X)
        y_train.append(1)

    index = rng.permutation(range(size))

    X_train = np.array(X_train)
    y_train = np.array(y_train)

    X_train = X_train[index, :, :, :]
    y_train = y_train[index]
    return X_train, y_train


def training_model(rng, d):
    shape = (d, d, 1)
    X_train, y_train = make_training_dataset(rng, size=1000, shape=shape)
    model = models.simple_model_classification(shape)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    model.fit(X_train, y_train, validation_split=0.3, epochs=100, verbose=1)
    model.save_weights(f"models/cam_d{d}.h5")


if __name__ == "__main__":
    rng = np.random.default_rng(seed=0)
    for d in [8, 16, 32, 64]:
        training_model(rng, d)
