import numpy as np
import tensorflow as tf
from layers import Residual


class AutoEncoder(tf.keras.Model):

    def __init__(self, cfg):
        super(AutoEncoder, self).__init__()
        filters = cfg.min_filters
        self.shape = (32, 32)

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
        
        self.decode.add(tf.keras.layers.Conv2D(cfg.num_channels, 1, 1, padding="same", activation=cfg.activ, kernel_initializer=cfg.init))
    
    def call(self, x_in, actions):
        encoded = self.encode(x_in)
        encoded = tf.concat([encoded, actions], axis=-1)
        x_out = self.decode(encoded)
        return x_out
