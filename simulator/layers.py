import tensorflow as tf

# Residual connection block, modified s.t. batch norm is after activation
class Residual(tf.keras.layers.Layer):

    def __init__(self, cfg, filters):
        super(Residual, self).__init__()
        self.layer = tf.keras.Sequential([
            tf.keras.layers.Conv2D(filters, 3, 1, padding="same", activation=cfg.activ, kernel_initializer=cfg.init),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv2D(filters, 3, 1, padding="same", activation=cfg.activ, kernel_initializer=cfg.init),
            tf.keras.layers.BatchNormalization()
        ])

    def call(self, x_in):
        x = self.layer(x_in)
        x_out = x + x_in
        return x_out
