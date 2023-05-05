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
        return tf.math.reduce_max(inputs, axis=1, keepdims=True)


class min_layer(keras.layers.Layer):
    def call(self, inputs):
        return tf.math.reduce_min(inputs, axis=1, keepdims=True)


class mean_layer(keras.layers.Layer):
    def call(self, inputs):
        return tf.math.reduce_max(inputs, axis=1, keepdims=True)/inputs.shape[1]


class concat_layer(keras.layers.Layer):
    def call(self, inputs):
        return tf.conact(inputs, axis=0, keepdims=True)


# create custom model
class CustomModel(keras.models.Model):
    def __init__(self):
        super().__init__()
        self.flatten = tf.keras.layers.Flatten()
        self.hidden1 = tf.keras.layers.Dense(256, "selu")
        self.hidden2 = tf.keras.layers.Dense(256, "selu")
        self.sum_layer = sum_layer()
        self.max_layer = max_layer()
        self.min_layer = min_layer()
        self.mean_layer = mean_layer()
        self.hidden3 = tf.keras.layers.Dense(256, "selu")
        self.op = tf.keras.layers.Dense(2, activation="softmax")

    def call(self, inputs, custom_layer_str):
        flatten = self.flatten(inputs)
        hidden1 = self.hidden1(flatten)
        hidden2 = self.hidden2(hidden1)
        if custom_layer_str == "Sum":
            custom_layer = self.sum_layer(hidden2)
        elif custom_layer_str == "Max":
            custom_layer = self.max_layer(hidden2)
        elif custom_layer_str == "Min":
            custom_layer = self.min_layer(hidden2)
        elif custom_layer_str == "Mean":
            custom_layer = self.mean_layer(hidden2)
        elif custom_layer_str == "Conact":
            custom_layer_sum = self.sum_layer(hidden2)
            custom_layer_max = self.max_layer(hidden2)
            custom_layer_min = self.min_layer(hidden2)
            custom_layer_mean = self.mean_layer(hidden2)
            custom_layer = self.conact_layer([custom_layer_sum, custom_layer_max,
            custom_layer_min, custom_layer_mean])
        hidden3 = self.hidden3(custom_layer)
        op = self.op(hidden3)
        return op

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
