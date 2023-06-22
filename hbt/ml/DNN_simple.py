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


class var_layer(keras.layers.Layer):
    def call(self, inputs):
        variance_inputs = tf.math.reduce_variance(inputs, axis=1, keepdims=True)
        return variance_inputs


class std_layer(keras.layers.Layer):
    def call(self, inputs):
        std_inputs = tf.math.reduce_std(inputs, axis=1, keepdims=True)
        return std_inputs


class concat_layer(keras.layers.Layer):
    def call(self, inputs):
        return tf.concat(inputs, axis=-1)


# create custom model
class CustomModel(keras.models.Model):
    def __init__(self, layer_strs, n_output_nodes, batch_norm_deepSets, batch_norm_ff,
                 nodes_deepSets, nodes_ff, activation_func_deepSets, activation_func_ff):
        super().__init__(self)

        # Set Parameters for the Model
        self.layer_strs = layer_strs
        self.n_output_nodes = n_output_nodes
        self.batch_norm_deepSets = batch_norm_deepSets
        self.batch_norm_ff = batch_norm_ff
        self.nodes_deepSets = nodes_deepSets
        self.nodes_ff = nodes_ff
        self.activation_func_deepSets = activation_func_deepSets
        self.activation_func_ff = activation_func_ff

        # Layers for the Model
        self.hidden1_deep = tf.keras.layers.Dense(nodes_deepSets[0], activation_func_deepSets)
        self.batch_hidden1_deep = tf.keras.layers.TimeDistributed(tf.keras.layers.BatchNormalization())

        self.hidden2_deep = tf.keras.layers.Dense(nodes_deepSets[1], activation_func_deepSets)
        self.batch_hidden2_deep = tf.keras.layers.TimeDistributed(tf.keras.layers.BatchNormalization())

        self.hidden3_deep = tf.keras.layers.Dense(nodes_deepSets[2], activation_func_deepSets)
        self.batch_hidden3_deep = tf.keras.layers.TimeDistributed(tf.keras.layers.BatchNormalization())

        self.hidden4_deep = tf.keras.layers.Dense(nodes_deepSets[3], activation_func_deepSets)
        self.batch_hidden4_deep = tf.keras.layers.TimeDistributed(tf.keras.layers.BatchNormalization())

        self.hidden5_deep = tf.keras.layers.Dense(nodes_deepSets[4], activation_func_deepSets)
        self.batch_hidden5_deep = tf.keras.layers.TimeDistributed(tf.keras.layers.BatchNormalization())

        self.hidden6_deep = tf.keras.layers.Dense(nodes_deepSets[5], activation_func_deepSets)

        # Deep Sets permuation invariant custom layers and concat Layer
        self.sum_layer = sum_layer()
        self.max_layer = max_layer()
        self.min_layer = min_layer()
        self.mean_layer = mean_layer()
        self.var_layer = var_layer()
        self.std_layer = std_layer()
        self.concat_layer = concat_layer()

        # Layers for the second half of the Network
        self.hidden1 = tf.keras.layers.Dense(nodes_ff[0], activation_func_ff)
        self.batch_hidden1 = tf.keras.layers.BatchNormalization(axis=-1)

        self.hidden2 = tf.keras.layers.Dense(nodes_ff[1], activation_func_ff)
        self.batch_hidden2 = tf.keras.layers.BatchNormalization(axis=-1)

        self.hidden3 = tf.keras.layers.Dense(nodes_ff[2], activation_func_ff)
        self.batch_hidden3 = tf.keras.layers.BatchNormalization(axis=-1)

        self.hidden4 = tf.keras.layers.Dense(nodes_ff[3], activation_func_ff)
        self.batch_hidden4 = tf.keras.layers.BatchNormalization(axis=-1)

        self.hidden5 = tf.keras.layers.Dense(nodes_ff[4], activation_func_ff)
        self.batch_hidden5 = tf.keras.layers.BatchNormalization(axis=-1)

        self.hidden6 = tf.keras.layers.Dense(nodes_ff[5], activation_func_ff)

        self.op = tf.keras.layers.Dense(n_output_nodes, activation="softmax")

    def call(self, inputs):
        inp_deepSets, inp2 = inputs

        # Deep Sets Arcitecture
        hidden1_deep = self.hidden1_deep(inp_deepSets)
        print('Hidden1 Deep Sets: ', hidden1_deep.shape)
        if self.batch_norm_deepSets:
            hidden1_deep = self.batch_hidden1_deep(hidden1_deep)
            print('Inputs Deep Sets: ', inp_deepSets.shape)

        hidden2_deep = self.hidden2_deep(hidden1_deep)
        if self.batch_norm_deepSets:
            hidden2_deep = self.batch_hidden2_deep(hidden2_deep)

        hidden3_deep = self.hidden3_deep(hidden2_deep)
        if self.batch_norm_deepSets:
            hidden3_deep = self.batch_hidden3_deep(hidden3_deep)

        hidden4_deep = self.hidden4_deep(hidden3_deep)
        if self.batch_norm_deepSets:
            hidden4_deep = self.batch_hidden4_deep(hidden4_deep)

        hidden5_deep = self.hidden5_deep(hidden4_deep)
        if self.batch_norm_deepSets:
            hidden5_deep = self.batch_hidden5_deep(hidden5_deep)

        hidden6_deep = self.hidden6_deep(hidden5_deep)

        # choose permuation invariant function for deep sets and concatenate
        layer_list = []
        for layer_str in self.layer_strs:
            if layer_str == "Sum":
                sum_layer = self.sum_layer(hidden6_deep)
                layer_list.append(sum_layer)
            elif layer_str == "Max":
                max_layer = self.max_layer(hidden6_deep)
                layer_list.append(max_layer)
            elif layer_str == "Min":
                min_layer = self.min_layer(hidden6_deep)
                layer_list.append(min_layer)
            elif layer_str == "Mean":
                mean_layer = self.mean_layer(hidden6_deep)
                layer_list.append(mean_layer)
            elif layer_str == "Var":
                var_layer = self.var_layer(hidden6_deep)
                layer_list.append(var_layer)
            elif layer_str == "Std":
                std_layer = self.std_layer(hidden6_deep)
                layer_list.append(std_layer)

        # Concatenation of layers and inp2 for FF
        layer_list.append(inp2)
        concat = self.concat_layer(layer_list)

        # second half of the network using Deep Sets output and event information
        hidden1 = self.hidden1(concat)
        print('Hidden1 Deep Sets Norm: ', hidden1.shape)
        if self.batch_norm_ff:
            hidden1 = self.batch_hidden1(hidden1)
            print('Hidden1 Batch: ', hidden1.shape)

        hidden2 = self.hidden2(hidden1)
        if self.batch_norm_ff:
            hidden2 = self.batch_hidden2(hidden2)

        hidden3 = self.hidden3(hidden2)
        if self.batch_norm_ff:
            hidden3 = self.batch_hidden3(hidden3)

        hidden4 = self.hidden4(hidden3)
        if self.batch_norm_ff:
            hidden4 = self.batch_hidden4(hidden4)

        hidden5 = self.hidden5(hidden4)
        if self.batch_norm_ff:
            hidden5 = self.batch_hidden5(hidden5)

        hidden6 = self.hidden6(hidden5)

        op = self.op(hidden6)

        # print output shapes of all layers

        print('Concat for next NN: ', concat.shape)
        print('Inputs 2: ', inp2.shape)
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
