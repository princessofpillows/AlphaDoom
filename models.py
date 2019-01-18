import tensorflow as tf
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
        self.block = tf.keras.Sequential([
            # Filters, Kernel Size, Strides
            tf.keras.layers.Conv2D(256, 3, 1, activation=cfg.activ, kernel_initializer=cfg.init),
            tf.keras.layers.BatchNormalization(),
            Residual(),
            Residual(),
            Residual(),
            Residual(),
            Residual(),
            Residual(),
            Residual(),
            Residual(),
            Residual(),
            Residual(),
            Residual(),
            Residual(),
            Residual(),
            Residual(),
            Residual(),
            Residual(),
            Residual(),
            Residual(),
            Residual()
        ])
        self.policy = Policy()
        self.value = Value()
    
    def call(self, x_in):
        x_in = self.block(x_in)
        x_p = self.policy(x_in)
        x_v = self.value(x_in)
        return x_p, x_v
        