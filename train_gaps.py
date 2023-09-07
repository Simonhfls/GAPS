import argparse
import os
import tensorflow as tf
import script.utils.io as utils
import script.train.Data as Data
from script.train.model import DeformationModel
from script.utils.configparser import config_parser
from shutil import copyfile
from script.utils.global_vars import ROOT_DIR
# import wandb      # uncomment if using wandb


def make_model(config):
    model = DeformationModel(config)
    print("Building model...")
    model.build(input_shape=[(None, None, 72), (None, None, 10), (None, None, 3), (None, None, 3), (None, None, 256)])

    model.summary()

    print("Compiling model...")
    optimizer = tf.keras.optimizers.Adam(learning_rate=config.getfloat('lr'))
    model.compile(optimizer=optimizer)
    return model


def lr_scheduler(epoch, lr):
    if epoch < 10:
        return lr
    else:
        return 0.0001


class EpochCallback(tf.keras.callbacks.Callback):
    def on_epoch_begin(self, epoch, logs=None):
        self.model.current_epoch.assign(epoch)
        tf.print('Set current_epoch to:', epoch)

    def on_epoch_end(self, epoch, logs=None):
        metrics = {k: v for k, v in logs.items()}
        # wandb.log({**metrics, "Epoch": epoch})        # uncomment if using wandb


def main(config):
    num_gpus = len(tf.config.list_physical_devices('GPU'))

    # Set Multi-gpu training
    if num_gpus > 1:
        mirrored_strategy = tf.distribute.MirroredStrategy()
        with mirrored_strategy.scope():
            model = make_model(config)
        batch_size = config.getint('batch_size_per_gpu') * num_gpus
    else:
        model = make_model(config)
        batch_size = config.getint('batch_size_per_gpu')

    print("Reading data...")
    window_size = config.getint('window_size')

    train_data = Data.Data(mode='train', window_size=window_size)
    validation_data = Data.Data(mode="validation")

    d_train = tf.data.Dataset.from_generator(train_data,
                                             output_signature=(
                                                 tf.TensorSpec(shape=(None, 72), dtype=tf.float32),
                                                 tf.TensorSpec(shape=(None, 10), dtype=tf.float32),
                                                 tf.TensorSpec(shape=(None, 3), dtype=tf.float32),
                                                 tf.TensorSpec(shape=(None, 3), dtype=tf.float32),
                                             ))
    d_train = d_train.batch(batch_size).repeat()
    d_train = d_train.prefetch(tf.data.AUTOTUNE).repeat()
    d_val = tf.data.Dataset.from_generator(validation_data,
                                           output_signature=(
                                               tf.TensorSpec(shape=(None, 72), dtype=tf.float32),
                                               tf.TensorSpec(shape=(None, 10), dtype=tf.float32),
                                               tf.TensorSpec(shape=(None, 3), dtype=tf.float32),
                                               tf.TensorSpec(shape=(None, 3), dtype=tf.float32),
                                           ))
    d_val = d_val.batch(batch_size)

    print("Training...")
    model.fit(
        d_train,
        validation_data=d_val,
        epochs=config.getint('num_epochs'),
        steps_per_epoch=train_data.__len__() // batch_size,
        validation_steps=validation_data.__len__() // batch_size,
        callbacks=[
            tf.keras.callbacks.ModelCheckpoint(
                filepath=os.path.join(log_dir, 'trained_model', '{epoch}'),
                save_freq='epoch',
                period=config.getint('i_save')
            ),
            tf.keras.callbacks.LearningRateScheduler(lr_scheduler),
            EpochCallback()
        ],
    )


if __name__ == '__main__':
    utils.check_gpus()
    for gpu in tf.config.experimental.list_physical_devices('GPU'):
        tf.config.experimental.set_memory_growth(gpu, True)

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default='config/gaps_tshirt.ini')
    opts = parser.parse_args()
    args_all = config_parser(os.path.join(ROOT_DIR, opts.config))
    config = args_all['DEFAULT']

    log_dir = os.path.join(ROOT_DIR, config['log_dir'])
    print('log dir:', log_dir)

    if tf.distribute.get_replica_context().replica_id_in_sync_group == 0:
        if not os.path.exists(log_dir):
            print('create dir:', log_dir)
            os.makedirs(log_dir)
        copyfile(os.path.join(ROOT_DIR, opts.config), os.path.join(log_dir, 'args.ini'))

        # # uncomment if using wandb
        # print("Wandb init ...")
        # wandb.init(
        #     project="GAPS",
        #     name=config['expt_name'],
        #     mode='offline',
        #     config={}
        # )
        # for key, value in config.items():
        #     wandb.config[key] = value

    main(config)
