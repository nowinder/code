from einops.layers.tensorflow import Rearrange
import tensorflow as tf
from tensorflow.keras import layers

from nd_mlp_mixer.mlp import MLP

class MLPMM(tf.keras.Model):
    '''use Rearrange not conv2D to make tokens and no'''
    def __init__(
            self,
            num_blocks,
            patch_size,
            tokens_mlp_dim,
            channels_mlp_dim,
            multis = [1,64,1,1],
            expand_axis = 1,
            filters_num = [16,8],
    ):
        super().__init__()
        self.make_tokens = Rearrange("b (h p1) (w p2) c -> b (h w) (p2 p1 c)", p1=patch_size, p2=patch_size)
        self.mixers = [
            NdMixerBlock([tokens_mlp_dim, channels_mlp_dim]) for _ in range(num_blocks)
        ]
        self.batchnorm = layers.BatchNormalization()
        self.backim = Rearrange(' b (h w) c -> b h w c',h = 8)
        self.convtrans = layers.Conv2DTranspose(2,16,16,activation='softmax')
        self.expanco = layers.Lambda(lambda x: tf.tile(tf.expand_dims(x, axis=expand_axis), multis))
        self.blos = [UpsaconvBlock(filters_num) for _ in range(1)]
        self.upsam = layers.UpSampling2D(size=(2,2), interpolation= 'bicubic')
        self.convlast = layers.Conv2D(2,1,activation = 'softmax', use_bias=False, padding='same')

        # self.clf = layers.Dense(num_classes, kernel_initializer="zeros",use_bias=False)

    def call(self, inputs):
        x = self.make_tokens(inputs)
        for mixer in self.mixers:
            x = mixer(x)
        x = self.expanco(x)
        x = self.batchnorm(x)
        for blo in self.blos:
            x = blo(x)
        x = self.upsam(x)
        x = self.convlast(x)
        return x

class MLPMixer(tf.keras.Model):
    """Original MLP-Mixer, with same API as paper."""

    def __init__(
        self,
        num_classes,
        num_blocks,
        patch_size,
        hidden_dim,
        tokens_mlp_dim,
        channels_mlp_dim,
        
    ):
        super().__init__()
        s = (patch_size, patch_size)
        self.make_tokens = layers.Conv2D(hidden_dim, s, s)
        self.rearrange = Rearrange("n h w c -> n (h w) c")
        self.mixers = [
            NdMixerBlock([tokens_mlp_dim, channels_mlp_dim]) for _ in range(num_blocks)
        ]
        self.batchnorm = layers.BatchNormalization()
        self.clf = layers.Dense(num_classes, kernel_initializer="zeros")

    def call(self, inputs):
        x = self.make_tokens(inputs)
        x = self.rearrange(x)
        for mixer in self.mixers:
            x = mixer(x)
        x = self.batchnorm(x)
        x = tf.reduce_mean(x, axis=1)
        return self.clf(x)


class NdMixerBlock(layers.Layer):
    "N-dimensional MLP-mixer block, same as paper when 2-dimensional."

    def __init__(self, mlp_dims: list = None, activation=tf.nn.gelu):
        super().__init__()
        self.mlp_dims = mlp_dims
        self.activation = activation

    def build(self, input_shape):
        ndim = len(input_shape) - 1
        mlp_dims = self.mlp_dims if self.mlp_dims else [None] * ndim

        self.mlps = [
            MLP(input_shape[i + 1], mlp_dims[i], axis=i + 1, activation=self.activation)
            for i in range(ndim)
        ]

        self.batchnorms = [layers.BatchNormalization() for _ in range(ndim)]

    def call(self, inputs):
        h = inputs
        for mlp, batchnorm in zip(self.mlps, self.batchnorms):
            h = h + mlp(batchnorm(h))
        return h
    
class UpsaconvBlock(layers.Layer):
    '''nn'''

    def __init__(self, filters_num: list = None, activation=tf.nn.gelu):
        super().__init__()
        self.filters_num = filters_num
        self.activation = activation

    def build(self,input_shape):
        ndim = len(self.filters_num)
        self.upsamlpes = [layers.UpSampling2D(size=(2,2), interpolation= 'bicubic')
                          for _ in range(ndim)]
        self.convs = [layers.Conv2D(self.filters_num[i], 2, 2, activation=self.activation)
                      for i in range(ndim)]
    def call(self, inputs):
        # h = layers.UpSampling2D(size=(2,2), interpolation= 'lanczos5')(inputs)
        h = inputs
        for conv, upsample in zip(self.convs,self.upsamlpes):
            h = layers.Concatenate()([h, conv(upsample(h))])
        return h
