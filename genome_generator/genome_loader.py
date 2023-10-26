import logging
import pathlib

import pysam

logger = logging.getLogger(__name__)


class NoGenomeDataFoundError(ValueError):
    pass


class GenomeLoader:
    ALPHABET = 'ACGT'

    def __init__(self, path, extension, batch_size=1):
        folder = pathlib.Path(path)
        self._filename_generator = folder.rglob("*." + extension)
        self._filename = "*." + extension
        self._offset = 0
        self._processed_files = []
        self._batch_size = batch_size

    def __call__(self):
        no_data_found = True
        for file in self._filename_generator:
            logger.debug(f"Found file {file}")
            logger.debug(f"Processed files: {self._processed_files}")
            logger.debug(f"Absolute path: {str(file.absolute())}")
            if file.is_file() and str(file.absolute()) not in self._processed_files:
                logger.debug(f"It's a file!")
                filename = file.name
                self._filename = file.absolute()
                self._offset = 0
                if filename.endswith('.cram'):
                    # file.seek(offset)
                    with pysam.AlignmentFile(file, 'rc') as input_file:
                        count = 0
                        row = []
                        for read in input_file:
                            count += 1
                            line = read.query_sequence
                            row.extend(self._to_numeric_list(line))
                            if count == self._batch_size:
                                yield [row]
                                count = 0
                                row = []
                                no_data_found = False
                                self._offset = input_file.tell()
                elif filename.endswith('.fasta'):
                    logger.debug(f"Reading fasta file")
                    with open(file) as input_file:
                        count = 0
                        row = []
                        for line in input_file:
                            count += 1
                            row.extend(self._to_numeric_list(line))
                            if count == self._batch_size:
                                yield [row]
                                count = 0
                                row = []
                                no_data_found = False
                else:
                    raise ValueError('Unknown file format')
                self._processed_files.append(str(file.absolute()))
                logger.debug(f"Added processed file {file.absolute()}")
        if no_data_found:
            pass
            # yield []
            # raise NoGenomeDataFoundError()

    def _to_numeric_list(self, line):
        row = list(filter(lambda x: x > 0, map(self._to_numeric, line)))
        row += [0.0] * (100 - len(row))
        return row

    def _to_numeric(self, c):
        i = self.ALPHABET.find(c)
        if i == -1 and c != 'N':
            logger.error('Unknown symbol: ' + c + '. Skipping.')
        return float(i) / (len(self.ALPHABET) - 1)

    def restore_position(self, position):
        self._processed_files = position[0].decode("utf-8").split("\n")
        self._filename = position[1].decode("utf-8")
        self._offset = int(position[2].decode("utf-8"))

    def current_position(self):
        return ["\n".join(self._processed_files), str(self._filename), str(self._offset)]
