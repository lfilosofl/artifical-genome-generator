from genome_generator.generator import GenomeGenerator


def main():
    generator = GenomeGenerator('data/models/genome_generator.keras')
    generator.generate_genomes_file(1000, 'data/output/result4.fasta')


if __name__ == '__main__':
    main()
