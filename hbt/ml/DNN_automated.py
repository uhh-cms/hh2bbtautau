import tensorflow as tf


# Define Custom Aggregation Functions for deep Sets

class SumLayer(tf.keras.layers.Layer):
    def call(self, inputs):
        sum_inputs = tf.math.reduce_sum(inputs, axis=1, keepdims=False)
        return sum_inputs


class MaxLayer(tf.keras.layers.Layer):
    def call(self, inputs):
        max_inputs = tf.math.reduce_max(inputs, axis=1, keepdims=False)
        return max_inputs


class MinLayer(tf.keras.layers.Layer):
    def call(self, inputs):
        min_inputs = tf.math.reduce_min(inputs, axis=1, keepdims=False)
        return min_inputs


class MeanLayer(tf.keras.layers.Layer):
    def call(self, inputs):
        mean_inputs = tf.math.reduce_sum(inputs, axis=1, keepdims=False) / inputs.shape[1]
        return mean_inputs


class VarLayer(tf.keras.layers.Layer):
    def call(self, inputs):
        variance_inputs = tf.math.reduce_variance(inputs, axis=1, keepdims=False)
        return variance_inputs


class StdLayer(tf.keras.layers.Layer):
    def call(self, inputs):
        std_inputs = tf.math.reduce_std(inputs, axis=1, keepdims=False)
        return std_inputs


class DenseBlock(tf.keras.layers.Layer):

    def __init__(self, nodes, activation, deepset=False):
        super(DenseBlock, self).__init__()
        self.deepset = deepset

        if self.deepset:
            layers = self.timedistributed_dense_block(nodes, activation)
        else:
            layers = self.dense_block(nodes, activation)
        self.dense, self.batchnorm, self.activation = layers

    def dense_block(self, nodes, activation):
        name_prefix = 'FeedForward/DenseBlock/'

        layers = (tf.keras.layers.Dense(nodes, use_bias=False,
                                        name=name_prefix + 'Dense'),
                  tf.keras.layers.BatchNormalization(
                      name=name_prefix + 'BatchNorm'),
                  tf.keras.layers.Activation(
                      activation, name=name_prefix + activation),
                  )
        return layers

    def timedistributed_dense_block(self, nodes, activation):
        name_prefix = 'DeepSet/DenseBlock/'

        layers = (tf.keras.layers.Dense(nodes, use_bias=False,
                                        name=name_prefix + 'Dense'),
                  tf.keras.layers.BatchNormalization(
                      name=name_prefix + 'BatchNorm'),
                  tf.keras.layers.Activation(
                      activation, name=name_prefix + activation)
                  )
        layers = [tf.keras.layers.TimeDistributed(layer) for layer in layers]
        return layers

    def call(self, inputs):
        x = self.dense(inputs)
        x = self.batchnorm(x)
        output = self.activation(x)
        return output


class DeepSet(tf.keras.Model):

    def __init__(self, nodes, activations, aggregations):
        super(DeepSet, self).__init__()
        self.aggregations = aggregations
        self.hidden_layers = [DenseBlock(node, activation, deepset=True)
                              for node, activation in zip(nodes, activations)]

        # aggregation layers
        self.sum_layer = SumLayer()
        self.max_layer = MaxLayer()
        self.min_layer = MinLayer()
        self.mean_layer = MeanLayer()
        self.var_layer = VarLayer()
        self.std_layer = StdLayer()
        self.concat_layer = tf.keras.layers.Concatenate(axis=1)

    def call(self, inputs):
        x = self.hidden_layers[0](inputs)
        for layer in self.hidden_layers[1:]:
            x = layer(x)

        aggregation_layers = []
        for func in self.aggregations:
            if func == "Sum":
                sum_layer = self.sum_layer(x)
                aggregation_layers.append(sum_layer)
            elif func == "Max":
                max_layer = self.max_layer(x)
                aggregation_layers.append(max_layer)
            elif func == "Min":
                min_layer = self.min_layer(x)
                aggregation_layers.append(min_layer)
            elif func == "Mean":
                mean_layer = self.mean_layer(x)
                aggregation_layers.append(mean_layer)
            elif func == "Var":
                var_layer = self.var_layer(x)
                aggregation_layers.append(var_layer)
            elif func == "Std":
                std_layer = self.std_layer(x)
                aggregation_layers.append(std_layer)

        output = self.concat_layer(aggregation_layers)

        return output


class FeedForwardNetwork(tf.keras.Model):

    def __init__(self, nodes, activations, n_classes):
        super(FeedForwardNetwork, self).__init__()
        self.hidden_layers = [DenseBlock(node, activation, deepset=False)
                              for node, activation in zip(nodes, activations)]
        self.output_layer = tf.keras.layers.Dense(n_classes, activation='softmax')

    def call(self, inputs):
        x = self.hidden_layers[0](inputs)
        for layer in self.hidden_layers[1:]:
            x = layer(x)
        output = self.output_layer(x)
        return output


class CombinedDeepSetNetwork(tf.keras.Model):
    def __init__(self, deepset_config, feedforward_config):
        super(CombinedDeepSetNetwork, self).__init__()

        self.deepset_network = DeepSet(**deepset_config)
        self.concat_layer = tf.keras.layers.Concatenate(axis=1)
        self.feed_forward_network = FeedForwardNetwork(**feedforward_config)

    def call(self, inputs):
        deepset_inputs, feedforward_inputs = inputs
        deepset_output = self.deepset_network(deepset_inputs)
        concatenated_inputs = self.concat_layer((deepset_output, feedforward_inputs))
        output = self.feed_forward_network(concatenated_inputs)
        return output


if __name__ == '__main__':
    deep_set_input = tf.ones(shape=(2, 3, 2))
    feed_forward_input = tf.ones(shape=(2, 32))

    deepset_config = {'nodes': (20, 10, 1), 'activations': ('relu', 'relu', 'selu')}
    feedforward_config = {'nodes': (128, 128, 128), 'activations': (
        'selu', 'selu', 'selu'), 'n_classes': 3}

    comb = CombinedDeepSetNetwork(deepset_config, feedforward_config)
    from IPython import embed
    embed()
