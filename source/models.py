import tensorflow as tf


def simple_model_classification(shape):
    inputs = tf.keras.layers.Input(shape=shape)
    conv1 = tf.keras.layers.Conv2D(4, (3, 3), padding="same", activation="relu")(inputs)
    conv2 = tf.keras.layers.Conv2D(4, (3, 3), padding="same", activation="relu")(conv1)
    maxpool1 = tf.keras.layers.MaxPool2D((2, 2))(conv2)
    conv3 = tf.keras.layers.Conv2D(4, (3, 3), padding="same", activation="relu")(
        maxpool1
    )
    conv4 = tf.keras.layers.Conv2D(4, (3, 3), padding="same", activation="relu")(conv3)
    up1 = tf.keras.layers.UpSampling2D((2, 2))(conv4)
    gap1 = tf.keras.layers.GlobalAveragePooling2D()(up1)
    dense1 = tf.keras.layers.Dense(1, activation="sigmoid")(gap1)

    model = tf.keras.Model(inputs=inputs, outputs=dense1)

    return model


class CAM(tf.keras.layers.Layer):
    def __init__(self, previous_layer, shape, mode="thr", thr=0, **kwargs):
        super(CAM, self).__init__(**kwargs)
        self.previous_layer = previous_layer
        self.cam_weights = tf.reshape(previous_layer.get_weights()[0], [-1])
        self.input_size = shape
        self.mode = mode
        self.thr = thr

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "previous_layer": self.previous_layer,
                "shape": self.input_size,
                "mode": self.mode,
                "thr": self.thr,
            }
        )

        return config

    def get_weights(self):
        return [self.cam_weights]

    def call(self, inputs, output_shape=None):
        conv_output = inputs[0]
        output = inputs[1]

        if output_shape is None:
            output_shape = [self.input_size, conv_output]

        cam = tf.reduce_sum(conv_output * self.cam_weights, axis=3)

        return cam, output

    def compute_output_shape(self, input_shape):
        return [self.input_size, input_shape]
