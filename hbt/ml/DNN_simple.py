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
    def __init__(self, custom_layer_str, n_output_nodes):
        super().__init__(self)
        # Deep Sets Layers
        self.custom_layer_str = custom_layer_str
        self.n_output_nodes = n_output_nodes

        self.hidden1_deep = tf.keras.layers.Dense(64, "selu")
        self.batch_hidden1_deep = tf.keras.layers.TimeDistributed(tf.keras.layers.BatchNormalization())

        self.hidden2_deep = tf.keras.layers.Dense(64, "selu")
        self.batch_hidden2_deep = tf.keras.layers.TimeDistributed(tf.keras.layers.BatchNormalization())

        self.hidden3_deep = tf.keras.layers.Dense(64, "selu")
        self.batch_hidden3_deep = tf.keras.layers.TimeDistributed(tf.keras.layers.BatchNormalization())

        self.hidden4_deep = tf.keras.layers.Dense(64, "selu")
        self.batch_hidden4_deep = tf.keras.layers.TimeDistributed(tf.keras.layers.BatchNormalization())

        self.hidden5_deep = tf.keras.layers.Dense(64, "selu")
        self.batch_hidden5_deep = tf.keras.layers.TimeDistributed(tf.keras.layers.BatchNormalization())

        self.hidden6_deep = tf.keras.layers.Dense(64, "selu")

        # Deep Sets permuation invariant custom layers and concat Layer
        self.sum_layer = sum_layer()
        self.max_layer = max_layer()
        self.min_layer = min_layer()
        self.mean_layer = mean_layer()
        self.concat_layer = concat_layer()

        # Layers for the second half of the Network
        self.hidden1 = tf.keras.layers.Dense(256, "selu")
        self.batch_hidden1 = tf.keras.layers.BatchNormalization(axis=-1)

        self.hidden2 = tf.keras.layers.Dense(256, "selu")
        self.batch_hidden2 = tf.keras.layers.BatchNormalization(axis=-1)

        self.hidden3 = tf.keras.layers.Dense(256, "selu")
        self.batch_hidden3 = tf.keras.layers.BatchNormalization(axis=-1)

        self.hidden4 = tf.keras.layers.Dense(256, "selu")
        self.batch_hidden4 = tf.keras.layers.BatchNormalization(axis=-1)

        self.hidden5 = tf.keras.layers.Dense(256, "selu")
        self.batch_hidden5 = tf.keras.layers.BatchNormalization(axis=-1)

        self.hidden6 = tf.keras.layers.Dense(256, "selu")

        self.op = tf.keras.layers.Dense(n_output_nodes, activation="softmax")

    def call(self, inputs):
        inp_deepSets, inp2 = inputs

        # Deep Sets Arcitecture
        hidden1_deep = self.hidden1_deep(inp_deepSets)
        batch_hidden1_deep = self.batch_hidden1_deep(hidden1_deep)

        hidden2_deep = self.hidden2_deep(batch_hidden1_deep)
        batch_hidden2_deep = self.batch_hidden2_deep(hidden2_deep)

        hidden3_deep = self.hidden3_deep(batch_hidden2_deep)
        batch_hidden3_deep = self.batch_hidden3_deep(hidden3_deep)

        hidden4_deep = self.hidden4_deep(batch_hidden3_deep)
        batch_hidden4_deep = self.batch_hidden4_deep(hidden4_deep)

        hidden5_deep = self.hidden5_deep(batch_hidden4_deep)
        batch_hidden5_deep = self.batch_hidden5_deep(hidden5_deep)

        hidden6_deep = self.hidden6_deep(batch_hidden5_deep)

        # chosse permuation invariant function for deep sets and concatenate
        if self.custom_layer_str == "Sum":
            custom_layer = self.sum_layer(hidden6_deep)
        elif self.custom_layer_str == "Max":
            custom_layer = self.max_layer(hidden6_deep)
        elif self.custom_layer_str == "Min":
            custom_layer = self.min_layer(hidden6_deep)
        elif self.custom_layer_str == "Mean":
            custom_layer = self.mean_layer(hidden6_deep)
        elif self.custom_layer_str == "Concat":
            custom_layer_sum = self.sum_layer(hidden6_deep)
            custom_layer_max = self.max_layer(hidden6_deep)
            custom_layer_min = self.min_layer(hidden6_deep)
            custom_layer_mean = self.mean_layer(hidden6_deep)
            custom_layer = self.concat_layer([custom_layer_sum, custom_layer_max,
            custom_layer_min, custom_layer_mean])
        concat = self.concat_layer([custom_layer, inp2])

        # second half of the network using Deep Sets output and event information
        hidden1 = self.hidden1(concat)
        batch_hidden1 = self.batch_hidden1(hidden1)

        hidden2 = self.hidden2(batch_hidden1)
        batch_hidden2 = self.batch_hidden2(hidden2)

        hidden3 = self.hidden3(batch_hidden2)
        batch_hidden3 = self.batch_hidden3(hidden3)

        hidden4 = self.hidden4(batch_hidden3)
        batch_hidden4 = self.batch_hidden4(hidden4)

        hidden5 = self.hidden5(batch_hidden4)
        batch_hidden5 = self.batch_hidden5(hidden5)

        hidden6 = self.hidden6(batch_hidden5)

        op = self.op(hidden6)

        # print output shapes of all layers
        print('Inputs Deep Sets: ', inp_deepSets.shape)
        print('Hidden1 Deep Sets: ', hidden1_deep.shape)
        print('Hidden1 Deep Sets Norm: ', batch_hidden1_deep.shape)
        print(f'{self.custom_layer_str} Layer: ', custom_layer.shape)
        print('Concat for next NN: ', concat.shape)
        print('Hidden1: ', hidden1.shape)
        print('Hidden1 Batch: ', batch_hidden1.shape)
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


class SimpleModel(keras.models.Model):
    def __init__(self, n_output_nodes):
        super().__init__(self)
        self.n_output_nodes = n_output_nodes
        self.hidden1 = tf.keras.layers.Dense(256, "selu")
        self.batch_norm_1 = tf.keras.layers.BatchNormalization(axis=-1)
        self.hidden2 = tf.keras.layers.Dense(256, "selu")
        self.batch_norm_2 = tf.keras.layers.BatchNormalization(axis=-1)
        self.hidden3 = tf.keras.layers.Dense(256, "selu")
        self.batch_norm_3 = tf.keras.layers.BatchNormalization(axis=-1)
        self.hidden4 = tf.keras.layers.Dense(256, "selu")
        self.batch_norm_4 = tf.keras.layers.BatchNormalization(axis=-1)
        self.hidden5 = tf.keras.layers.Dense(256, "selu")
        self.batch_norm_5 = tf.keras.layers.BatchNormalization(axis=-1)
        self.hidden6 = tf.keras.layers.Dense(256, "selu")
        self.op = tf.keras.layers.Dense(n_output_nodes, activation="softmax")

    def call(self, inputs):
        input = inputs
        hidden1 = self.hidden1(input)
        batch_hidden1 = self.batch_norm_1(hidden1)
        hidden2 = self.hidden2(batch_hidden1)
        batch_hidden2 = self.batch_norm_2(hidden2)
        hidden3 = self.hidden3(batch_hidden2)
        batch_hidden3 = self.batch_norm_3(hidden3)
        hidden3 = self.hidden3(batch_hidden2)
        batch_hidden3 = self.batch_norm_3(hidden3)
        hidden4 = self.hidden4(batch_hidden3)
        batch_hidden4 = self.batch_norm_4(hidden4)
        hidden5 = self.hidden5(batch_hidden4)
        batch_hidden5 = self.batch_norm_5(hidden5)
        hidden6 = self.hidden6(batch_hidden5)
        op = self.op(hidden6)
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
