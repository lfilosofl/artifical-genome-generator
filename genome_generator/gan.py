import tensorflow as tf


class GenerativeAdversarialNetwork(tf.keras.Model):

    def __init__(self, generator, discriminator, loss_function, latent_dim):
        super().__init__()

        self.generator = generator
        self.discriminator = discriminator
        self.loss_fn = loss_function
        self.latent_dim = latent_dim

        self.discriminator_loss_metric = tf.keras.metrics.Mean(name="discriminator_loss")
        self.generator_loss_metric = tf.keras.metrics.Mean(name="generator_loss")

    @property
    def metrics(self):
        return [self.discriminator_loss_metric, self.generator_loss_metric]

    def train_step(self, data):
        batch_size = tf.shape(data)[0]
        noise = tf.random.normal([batch_size, self.latent_dim])

        with tf.GradientTape() as generator_tape, tf.GradientTape() as discriminator_tape:
            generated_data = self.generator.model(noise, training=True)

            real_output = self.discriminator.model(data, training=True)
            fake_output = self.discriminator.model(generated_data, training=True)

            generator_loss = self.generator_loss(fake_output)
            discriminator_loss = self.discriminator_loss(real_output, fake_output)

        gradients_of_generator = generator_tape.gradient(generator_loss, self.generator.model.trainable_variables)
        gradients_of_discriminator = discriminator_tape.gradient(discriminator_loss,
                                                                 self.discriminator.model.trainable_variables)

        self.__apply_gradients(self.generator, gradients_of_generator)
        self.__apply_gradients(self.discriminator, gradients_of_discriminator)

        self.discriminator_loss_metric.update_state(generator_loss)
        self.generator_loss_metric.update_state(discriminator_loss)

        return self.metrics_result()

    def discriminator_loss(self, real_output, fake_output):
        real_loss = self.loss_fn(tf.ones_like(real_output), real_output)
        fake_loss = self.loss_fn(tf.zeros_like(fake_output), fake_output)
        return real_loss + fake_loss

    def generator_loss(self, fake_output):
        return self.loss_fn(tf.ones_like(fake_output), fake_output)

    @staticmethod
    def __apply_gradients(network, gradients):
        network.optimizer.apply_gradients(zip(gradients, network.model.trainable_variables))

    def metrics_result(self):
        return {metric.name: metric.result() for metric in self.metrics}
