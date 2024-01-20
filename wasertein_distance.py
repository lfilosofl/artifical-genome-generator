from numpy.random import randn
from numpy.random import lognormal
from numpy.random import randint
from numpy.random import poisson
from scipy.stats import wasserstein_distance

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
    print(f"distance to valid data = {wasserstein_distance(generated_data, validation_data)}")
    uniform_values = randint(0, 3, n) / 3
    print(f"distance to uniform data = {wasserstein_distance(generated_data, uniform_values)}")
    print(f"distance to poisson data = {wasserstein_distance(generated_data, poisson(1, n))}")
    print(f"distance to normal data = {wasserstein_distance(generated_data, randn(n))}")
    print(f"distance to lognormal data = {wasserstein_distance(generated_data, lognormal(0, 1, n))}")
