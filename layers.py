import tensorflow as tf

# Network in network for googlenet
class Inception(tf.keras.layers.Layer):
    # _r serves as "reduction" of computational complexity for following layer
    # _p occurs following a pool layer
    def __init__(self, cfg, shp_1x1, shp_3x3_r, shp_3x3, shp_5x5_r, shp_5x5, shp_1x1_p):
        super(Inception, self).__init__()
        self.conv1x1 = tf.keras.layers.Conv2D(shp_1x1, 1, padding="same", activation=cfg.activ, kernel_initializer=cfg.init)

        self.conv3x3_r = tf.keras.layers.Conv2D(shp_3x3_r, 1, padding="same", activation=cfg.activ, kernel_initializer=cfg.init)
        self.conv3x3 = tf.keras.layers.Conv2D(shp_3x3, 3, padding="same", activation=cfg.activ, kernel_initializer=cfg.init)
        
        self.conv5x5_r = tf.keras.layers.Conv2D(shp_5x5_r, 1, padding="same", activation=cfg.activ, kernel_initializer=cfg.init)
        self.conv5x5 = tf.keras.layers.Conv2D(shp_5x5, 5, padding="same", activation=cfg.activ, kernel_initializer=cfg.init)

        self.pool3x3 = tf.keras.layers.MaxPool2D(3, 1, padding="same")
        self.conv1x1_p = tf.keras.layers.Conv2D(shp_1x1_p, 1, padding="same", activation=cfg.activ, kernel_initializer=cfg.init)

        self.concat = tf.keras.layers.Concatenate()
    
    def call(self, x_in):
        block0 = self.conv1x1(x_in)

        block1 = self.conv3x3_r(x_in)
        block1 = self.conv3x3(block1)
        
        block2 = self.conv5x5_r(x_in)
        block2 = self.conv5x5(block1)

        block3 = self.pool3x3(x_in)
        block3 = self.conv1x1_p(block3)

        x_out = self.concat([block0, block1, block2, block3])
        return x_out

# Used to help backpropogation from middle layers of googlenet
class Auxiliary(tf.keras.layers.Layer):
    
    def __init__(self, cfg):
        super(Auxiliary, self).__init__()
        self.layer = tf.keras.Sequential([
            tf.keras.layers.AveragePooling2D(5, 3),
            tf.keras.layers.Conv2D(128, 1, padding="same", activation=cfg.activ, kernel_initializer=cfg.init),
            tf.keras.layers.Dense(1024, activation=cfg.activ, kernel_initializer=cfg.init),
            # Perform dropout to reduce overfitting
            tf.keras.layers.Dropout(0.7),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(len(cfg.classes), kernel_initializer=cfg.init)
        ])

    def call(self, x_in):
        x_out = self.layer(x_in)
        return x_out

# Residual connection block, modified s.t. batch norm is after activation
class Residual(tf.keras.layers.Layer):

    def __init__(self, cfg):
        super(Residual, self).__init__()
        self.layer = tf.keras.Sequential([
            tf.keras.layers.Conv2D(256, 3, 1, activation=cfg.activ, kernel_initializer=cfg.init),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv2D(256, 3, 1, activation=cfg.activ, kernel_initializer=cfg.init),
            tf.keras.layers.BatchNormalization()
        ])

    def call(self, x_in):
        x = self.layer(x_in)
        x_out = x + x_in
        return x_out

# Policy head output for AlphaGoZero
class Policy(tf.keras.layers.Layer):

    def __init__(self, cfg):
        super(Policy, self).__init__()
        self.layer = tf.keras.Sequential([
            tf.keras.layers.Conv2D(2, 2, 1, activation=cfg.activ, kernel_initializer=cfg.init),
            tf.keras.layers.BatchNormalization(),
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
            tf.keras.layers.Conv2D(1, 1, 1, activation=cfg.activ, kernel_initializer=cfg.init),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(256, activation=cfg.activ, kernel_initializer=cfg.init),
            tf.keras.layers.Dense(1, activation=tf.keras.activations.tanh, kernel_initializer=cfg.init)
        ])

    def call(self, x_in):
        x_out = self.layer(x_in)
        return x_out
