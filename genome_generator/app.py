import sys

from genome_generator.gan import GenerativeAdversarialNetwork
from genome_generator.gan_monitor import CheckPointManager
from genome_generator.genome_loader import GenomeLoader, NoGenomeDataFoundError
from genome_generator.models import Generator, Discriminator
import tensorflow as tf


class TrainApp:

    def __init__(self):
        self.latent_size = 600
        self.epochs = 1
        self.noise_dim = 100
        self.reads_count = 10
        self.read_size = 100
        self.batch_size = self.reads_count * self.read_size
        self.generator = Generator(self.latent_size, self.batch_size)
        self.discriminator = Discriminator(self.batch_size)
        self.genome_loader = GenomeLoader("data/dataset", "cram", self.reads_count)
        self.dataset = tf.data.Dataset.from_generator(self.genome_loader, tf.float32, ([1, self.batch_size]))

        self.loss_function = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        self.gan = GenerativeAdversarialNetwork(self.generator, self.discriminator, self.loss_function, self.noise_dim)
        self.checkpoint_manager = CheckPointManager('data/checkpoints', 'ckpt', self.genome_loader, True)

    def train(self):
        self.gan.compile()
        try:
            self.gan.fit(self.dataset, epochs=self.epochs, callbacks=[self.checkpoint_manager])
            self.generator.model.save('data/models/genome_generator.keras')
        except NoGenomeDataFoundError:
            print('No data found')
        # except ValueError:
        #     print('Value error')
