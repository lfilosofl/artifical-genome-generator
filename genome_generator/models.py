import tensorflow as tf


class Discriminator:

    def __init__(self, input_size):
        self.__model = tf.keras.Sequential([
            tf.keras.Input(shape=(input_size,)),
            tf.keras.layers.Dense(input_size // 2, kernel_regularizer=tf.keras.regularizers.l2(0.0001)),
            tf.keras.layers.LeakyReLU(alpha=0.02),
            tf.keras.layers.Dense(input_size // 3, kernel_regularizer=tf.keras.regularizers.l2(0.0001)),
            tf.keras.layers.LeakyReLU(alpha=0.02),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ], name='ag_discriminator')

        self.__optimizer = tf.keras.optimizers.Adam(1e-4)

    @property
    def model(self):
        return self.__model

    @property
    def optimizer(self):
        return self.__optimizer


class Generator:

    def __init__(self, input_size, output_size):
        self.__model = tf.keras.Sequential([
            # tf.keras.Input(shape=(input_size,)),
            tf.keras.layers.Dense(int(output_size // 1.2), kernel_regularizer=tf.keras.regularizers.l2(0.0001)),
            tf.keras.layers.LeakyReLU(alpha=0.02),
            tf.keras.layers.Dense(int(output_size // 1.1), kernel_regularizer=tf.keras.regularizers.l2(0.0001)),
            tf.keras.layers.LeakyReLU(alpha=0.02),
            tf.keras.layers.Dense(output_size, activation='tanh')
        ], name='ag_generator')

        self.__optimizer = tf.keras.optimizers.Adam(1e-4)

    @property
    def model(self):
        return self.__model

    @property
    def optimizer(self):
        return self.__optimizer
