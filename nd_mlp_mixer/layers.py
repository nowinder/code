import tensorflow as tf
# 自定义实现layer实例化和残差层，但是没有用到？
# residuallayer add in nd_mixer

class Layers(tf.keras.layers.Layer):
    def __init__(self, num_layers, make_layer):
        super().__init__()
        self.layers = [make_layer() for _ in range(num_layers)]

    def call(self, inputs):
        h = inputs
        for layer in self.layers:
            h = layer(h)
        return h


class ResidualLayers(tf.keras.layers.Layer):
    def __init__(self, num_layers, make_layer):
        super().__init__()
        self.layers = [make_layer() for _ in range(num_layers)]

    def call(self, inputs):
        h = inputs
        for layer in self.layers:
            h = h + layer(h)
        return h
