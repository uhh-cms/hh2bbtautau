import datetime
import os
from functools import wraps
import time
from columnflow.util import maybe_import

np = maybe_import("numpy")
ak = maybe_import("awkward")
tf = maybe_import("tensorflow")
pickle = maybe_import("pickle")
keras = maybe_import("tensorflow.keras")


class sum_layer(tf.keras.layers.Layer):
    def call(self, inputs):
        return tf.math.reduce_sum(inputs, axis=1, keepdims=True)


class max_layer(keras.layers.Layer):
    def call(self, inputs):
        max_inputs = tf.math.reduce_max(inputs, axis=1, keepdims=True)
        return max_inputs


class min_layer(keras.layers.Layer):
    def call(self, inputs):
        max_inputs = tf.math.reduce_min(inputs, axis=1, keepdims=True)
        return max_inputs


class mean_layer(keras.layers.Layer):
    def call(self, inputs):
        return tf.math.reduce_sum(inputs, axis=1, keepdims=True) / inputs.shape[1]


class concat_layer(keras.layers.Layer):
    def call(self, inputs):
        return tf.concat(inputs, axis=-1)


# create custom model
class CustomModel(keras.models.Model):
    def __init__(self, custom_layer_str):
        super().__init__(self)
        self.custom_layer_str = custom_layer_str
        self.hidden1 = tf.keras.layers.Dense(256, "selu")
        self.hidden2 = tf.keras.layers.Dense(256, "selu")
        self.sum_layer = sum_layer()
        self.max_layer = max_layer()
        self.min_layer = min_layer()
        self.mean_layer = mean_layer()
        self.concat_layer = concat_layer()
        self.hidden3 = tf.keras.layers.Dense(256, "selu")
        self.op = tf.keras.layers.Dense(2, activation="softmax")

    def call(self, inputs):
        inputs_deepSets, inputs_2 = inputs
        hidden1 = self.hidden1(inputs_deepSets)
        hidden2 = self.hidden2(hidden1)
        if self.custom_layer_str == "Sum":
            custom_layer = self.sum_layer(hidden2)
        elif self.custom_layer_str == "Max":
            custom_layer = self.max_layer(hidden2)
        elif self.custom_layer_str == "Min":
            custom_layer = self.min_layer(hidden2)
        elif self.custom_layer_str == "Mean":
            custom_layer = self.mean_layer(hidden2)
        elif self.custom_layer_str == "Concat":
            custom_layer_sum = self.sum_layer(hidden2)
            custom_layer_max = self.max_layer(hidden2)
            custom_layer_min = self.min_layer(hidden2)
            custom_layer_mean = self.mean_layer(hidden2)
            custom_layer = self.concat_layer([custom_layer_sum, custom_layer_max,
            custom_layer_min, custom_layer_mean])
        concat_next_nn = self.concat_layer([custom_layer, inputs_2])
        hidden3 = self.hidden3(concat_next_nn)
        op = self.op(hidden3)

        # print output shapes of all layers
        print('Inputs Deep Sets: ', inputs_deepSets.shape)
        print('Hidden1: ', hidden1.shape)
        print('Hidden2: ', hidden2.shape)
        print(f'{self.custom_layer_str} Layer: ', custom_layer.shape)
        print('Concat for next NN: ', concat_next_nn.shape)
        print('Hidden3: ', hidden3.shape)
        print('Output: ', op.shape)
        return op

    def model_deepsets(self):
        x = tf.keras.layers.Input(shape=(None, 1, 2, 4))
        return tf.keras.Model(inputs=[x], outputs=self.call(x))

    @tf.function
    def train_step(self, data):
        # Unpack the data. Its structure depends on your model and
        # on what you pass to `fit()`.
        x, y = data

        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)  # Forward pass
            # Compute the loss value
            # (the loss function is configured in `compile()`)
            loss = self.compiled_loss(y, y_pred)

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        # Update metrics (includes the metric that tracks the loss)
        self.compiled_metrics.update_state(y, y_pred)
        # Return a dict mapping metric names to current value
        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        # Unpack the data
        x, y = data
        # Compute predictions
        y_pred = self(x, training=False)
        # Updates the metrics tracking the loss
        self.compiled_loss(y, y_pred)
        # Update the metrics.
        self.compiled_metrics.update_state(y, y_pred)
        # Return a dict mapping metric names to current value.
        # Note that it will include the loss (tracked in self.metrics).
        return {m.name: m.result() for m in self.metrics}
