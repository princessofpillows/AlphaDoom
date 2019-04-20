import tensorflow as tf


# Residual connection block, modified s.t. batch norm is after activation
class Residual_Custom(tf.keras.layers.Layer):

    def __init__(self, cfg, filters):
        super(Residual_Custom, self).__init__()
        self.layer = tf.keras.Sequential([
            tf.keras.layers.Conv2D(filters, 3, 1, activation=cfg.activ, padding="same", kernel_initializer=cfg.init),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv2D(filters, 3, 1, activation=cfg.activ, padding="same", kernel_initializer=cfg.init),
            tf.keras.layers.BatchNormalization()
        ])

    def call(self, x_in):
        x = self.layer(x_in)
        x_out = x + x_in
        return x_out

# Residual connection block, modified s.t. batch norm is after activation
class Residual(tf.keras.layers.Layer):

    def __init__(self, cfg, filters):
        super(Residual, self).__init__()
        self.layer = tf.keras.Sequential([
            tf.keras.layers.Conv2D(filters, 3, 1, padding="same", kernel_initializer=cfg.init),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Conv2D(filters, 3, 1, padding="same", kernel_initializer=cfg.init),
            tf.keras.layers.BatchNormalization()
        ])

        self.ReLU = tf.keras.layers.ReLU()

    def call(self, x_in):
        x = self.layer(x_in)
        x_out = x + x_in
        return self.ReLU(x_out)


# Policy head output for AlphaGoZero
class Policy(tf.keras.layers.Layer):

    def __init__(self, cfg):
        super(Policy, self).__init__()
        self.layer = tf.keras.Sequential([
            tf.keras.layers.Conv2D(2, 1, kernel_initializer=cfg.init),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(len(cfg.actions), kernel_initializer=cfg.init)
        ])

    def call(self, x_in):
        x_out = self.layer(x_in)
        return x_out


# Value head output for AlphaGoZero
class Value(tf.keras.layers.Layer):

    def __init__(self, cfg):
        super(Value, self).__init__()
        self.layer = tf.keras.Sequential([
            tf.keras.layers.Conv2D(1, 1, kernel_initializer=cfg.init),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(256, activation="relu", kernel_initializer=cfg.init),
            tf.keras.layers.Dense(1, activation="tanh", kernel_initializer=cfg.init)
        ])

    def call(self, x_in):
        x_out = self.layer(x_in)
        return x_out
