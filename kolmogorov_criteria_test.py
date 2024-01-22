from scipy.stats import ks_2samp

from genome_generator.generator import GenomeGenerator
from genome_generator.genome_loader import GenomeLoader


def normalize(value):
    return round(((value + 1) * 3) / 2) / 3


if __name__ == '__main__':
    n = 10000
    generator = GenomeGenerator('data/models/genome_generator.keras')
    generated_data = generator.generate_genomes(n).flatten()
    generated_data = [normalize(value) for value in generated_data]
    loader = GenomeLoader("data/dataset", "cram", 10)
    loader_iterator = loader()
    row = []
    for i in range(n):
        row.extend(next(loader_iterator))
    validation_data = [x for xs in row for x in xs]
    ks_result = ks_2samp(generated_data, validation_data)
    print(f"pvalue = {ks_result.pvalue}")