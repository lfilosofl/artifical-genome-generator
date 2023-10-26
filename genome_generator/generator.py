import logging

import tensorflow as tf

logger = logging.getLogger(__name__)


class GenomeGenerator:

    def __init__(self, model_filename):
        self.model = tf.keras.models.load_model(model_filename)
        self.latent_size = self.model.layers[0].input_shape[1]

    def generate_genomes_file(self, amount, filename):
        logger.info("Start generation to " + filename + ", amount = " + str(amount))
        with open(filename, 'w') as output_file:
            genome = self._generate_genomes(amount)
            output_file.write('\n'.join(genome))

    def _generate_genomes(self, amount):
        noise = tf.random.normal([amount, self.latent_size])
        generated_data = self.model(noise, training=False)
        genome = self._convert_sequences_to_symbolic_form(generated_data.numpy())
        return genome

    @staticmethod
    def _convert_sequences_to_symbolic_form(sequences):
        # def linear_conversion(value, a, b, old_a, old_b): return (((value - old_a) * (b - a)) / (old_b - old_a)) + a

        def sequence_element_converter(value): return 'ACGT'[round(((value + 1) * 3) / 2)]

        return [''.join(map(sequence_element_converter, sequence)) for sequence in sequences]


class SequenceConverter:
    ALPHABET = 'ACGT'

    def float_to_symbolic(self, sequence):
        def linear_conversion(value, a, b, old_a, old_b): return (((value - old_a) * (b - a)) / (old_b - old_a)) + a

        def sequence_element_converter(value): return self.ALPHABET[round(((value + 1) * 3) / 2)]

        return ''.join(map(sequence_element_converter, sequence))

    def symbolic_to_float(self, sequence):
        def _to_numeric(c):
            i = self.ALPHABET.find(c)
            if i == -1 and c != 'N':
                logger.error('Unknown symbol: ' + c + '. Skipping.')
            return float(i) / (len(self.ALPHABET) - 1)
        row = list(filter(lambda x: x > 0, map(_to_numeric, sequence)))
        row += [0.0] * (100 - len(row))
        return row

