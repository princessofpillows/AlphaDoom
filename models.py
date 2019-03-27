import tensorflow as tf
import numpy as np
from layers import Residual, Policy, Value

# 19 residual connection alphagozero
class AlphaGoZero(tf.keras.Model):

    def __init__(self, cfg):
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
        
        
class Simulator(tf.keras.Model):

    def __init__(self, cfg):
        super(Simulator, self).__init__()
        filters = cfg.min_filters
        self.shape = cfg.resolutions["simulator"]

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
        
        self.decode.add(tf.keras.layers.Conv2D(cfg.num_channels, 1, 1, padding="same", kernel_initializer=cfg.init))
    
    def call(self, x_in, actions):
        encoded = self.encode(x_in)
        encoded = tf.concat([encoded, actions], axis=-1)
        x_out = self.decode(encoded)
        return x_out
    
    def predict(self, s0, action):
        s0_n = tf.image.per_image_standardization(s0)
        logits = self.call(s0_n[None], action[None])
        return logits + s0_n
