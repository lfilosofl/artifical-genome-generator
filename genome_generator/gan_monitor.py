import logging
import os

import tensorflow as tf

logger = logging.getLogger(__name__)


class CheckPointManager(tf.keras.callbacks.Callback):

    def __init__(self, checkpoint_dir, checkpoint_prefix, genome_loader, continue_from_last_checkpoint=True):
        super().__init__()
        self.continue_from_last_checkpoint = continue_from_last_checkpoint
        self.checkpoint = None
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_prefix = os.path.join(checkpoint_dir, checkpoint_prefix)
        self.checkpoint_manager = None
        self.genome_loader = genome_loader
        self.dataset_position = tf.Variable(genome_loader.current_position())

    def on_epoch_end(self, epoch, logs=None):
        self.dataset_position.assign(self.genome_loader.current_position())
        save_path = self.checkpoint_manager.save()
        logger.debug('Saved checkpoint: %s' % save_path)

    def on_batch_end(self, batch, logs=None):
        if (batch + 1) % 5000 == 0:
            self.dataset_position.assign(self.genome_loader.current_position())
            save_path = self.checkpoint_manager.save()
            logger.debug('Saved checkpoint: %s' % save_path)
            self.model.generator.model.save('data/models/genome_generator.keras')

    def on_train_begin(self, logs=None):
        self.checkpoint = tf.train.Checkpoint(generator_optimizer=self.model.generator.optimizer,
                                              discriminator_optimizer=self.model.discriminator.optimizer,
                                              generator=self.model.generator.model,
                                              discriminator=self.model.discriminator.model,
                                              dataset_position=self.dataset_position)
        self.checkpoint_manager = tf.train.CheckpointManager(self.checkpoint, self.checkpoint_dir, max_to_keep=3)
        if self.continue_from_last_checkpoint:
            self.checkpoint.restore(self.checkpoint_manager.latest_checkpoint).expect_partial()
            if self.checkpoint_manager.latest_checkpoint:
                restored_position = self.dataset_position.numpy().tolist()
                self.genome_loader.restore_position(restored_position)
                logger.debug(
                    f"Restored from {self.checkpoint_manager.latest_checkpoint}. Position: {restored_position}")
            else:
                logger.debug("Initializing from scratch.")
