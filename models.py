import tensorflow as tf
import numpy as np
from layers import Residual, Policy, Value

class Atari(tf.keras.Model):

    def __init__(self, cfg, num_classes):
        super(Atari, self).__init__()
        self.shape = (84, 84)
        self.block = tf.keras.Sequential([
            # Filters, Kernel Size, Strides
            tf.keras.layers.Conv2D(16, 8, 4, activation=cfg.activ, kernel_initializer=cfg.init),
            tf.keras.layers.Conv2D(32, 4, 2, activation=cfg.activ, kernel_initializer=cfg.init),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(256, activation=cfg.activ, kernel_initializer=cfg.init)
        ])
        self.out = tf.keras.layers.Dense(num_classes, kernel_initializer=cfg.init)
    
    def call(self, x_in):
        x_in = self.block(x_in)
        x_out = self.out(x_in)
        return x_out, [0]
    
    def test(self, x_in):
        feature = self.block(x_in)
        x_out = self.out(feature)
        return x_out, feature


# 19 residual connection alphagozero
class AlphaGoZero(tf.keras.Model):

    def __init__(self, cfg, num_classes):
        super(AlphaGoZero, self).__init__()
        self.shape = (19, 19)
        self.block = tf.keras.Sequential()
        self.block.add(tf.keras.layers.Conv2D(256, 3, 1, activation=cfg.activ, kernel_initializer=cfg.init))
        self.block.add(tf.keras.layers.BatchNormalization())
        for blk in range(19):
            self.block.add(Residual(cfg, 256))
        self.policy = Policy(cfg)
        self.value = Value(cfg)
    
    def call(self, x_in):
        x_in = self.block(x_in)
        x_p = self.policy(x_in)
        x_v = self.value(x_in)
        return x_p, x_v
        
        
class AutoEncoder(tf.keras.Model):

    def __init__(self, cfg):
        super(AutoEncoder, self).__init__()
        filters = cfg.min_filters
        self.shape = cfg.resolutions["autoencoder"]

        self.encode = tf.keras.models.Sequential()
        self.encode.add(tf.keras.layers.Conv2D(filters, 1, 1, padding="same", activation=cfg.activ, kernel_initializer=cfg.init))
        # Gets n from 2^n = shape
        num_layers = int(np.log2(self.shape[0])) + 1
        # From shape (nxn) -> (1x1)
        for i in range(num_layers):
            for blk in range(cfg.num_blks):
                self.encode.add(Residual(cfg, filters))
            if filters != cfg.max_filters:
                filters *= 2
            if i < num_layers - 1:
                self.encode.add(tf.keras.layers.Conv2D(filters, 3, 2, padding="same", activation=cfg.activ, kernel_initializer=cfg.init))
        
        self.decode = tf.keras.models.Sequential()
        self.decode.add(tf.keras.layers.Conv2D(filters, 1, 1, padding="same", activation=cfg.activ, kernel_initializer=cfg.init))
        # From shape (1x1) -> (nxn)
        for i in range(num_layers):
            for blk in range(cfg.num_blks):
                self.decode.add(Residual(cfg, filters))
            if filters != cfg.min_filters and i > 1:
                filters //= 2
            if i < num_layers - 1:
                self.decode.add(tf.keras.layers.Conv2DTranspose(filters, 3, 2, padding="same", activation=cfg.activ, kernel_initializer=cfg.init))
        
        self.decode.add(tf.keras.layers.Conv2D(cfg.output_channels, 1, 1, padding="same", kernel_initializer=cfg.init))
    
    def call(self, x_in, actions):
        encoded = self.encode(x_in)
        encoded = tf.concat([encoded, actions], axis=-1)
        x_out = self.decode(encoded)
        return x_out
