import os
import sys
import random
import tensorflow as tf
import numpy as np
import script.utils.io as utils
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/..')


class Data:
    def __init__(self, mode='train', window_size=3):
        # Read sample list
        self._pose_path = "assets/CMU"
        self.txt = f"assets/{mode}_seq.txt"
        self.read_txt()
        print('Loading dataset:', mode)
        self.mode = mode
        self.window_size = window_size
        self._poses, self._trans, self._trans_vel = self._get_pose()
        if self._poses.dtype == np.float64:
            self._poses = np.float32(self._poses)

        if self._trans.dtype == np.float64:
            self._trans = np.float32(self._trans)

        if self._trans_vel.dtype == np.float64:
            self._trans_vel = np.float32(self._trans_vel)

        self._n_samples = self._poses.shape[0]
        print('n_samples:', self._n_samples)

        self.on_epoch_end()

    def _get_pose(self):
        poses_array = np.zeros((1, 72))
        trans_array = np.zeros((1, 3))
        trans_vel_array = np.zeros((1, 3))

        for file_path in self.sequences:
            poses, trans, trans_vel = utils.load_motion(file_path, window_size=self.window_size)

            if poses.shape[0] > 300:        # take maximum 300 frames per sequence
                start = (poses.shape[0] - 300) // 2
                end = start + 300
                poses = poses[start:end]
                trans = trans[start:end]
                trans_vel = trans_vel[start:end]

            poses_array = np.concatenate((poses_array, poses), axis=0)
            trans_array = np.concatenate((trans_array, trans), axis=0)
            trans_vel_array = np.concatenate((trans_vel_array, trans_vel), axis=0)

        poses_array = np.delete(poses_array, 0, axis=0)
        trans_array = np.delete(trans_array, 0, axis=0)
        trans_vel_array = np.delete(trans_vel_array, 0, axis=0)

        print('total number of poses:', poses_array.shape[0])
        remainder = poses_array.shape[0] % self.window_size
        poses_array = np.delete(poses_array, range(poses_array.shape[0] - remainder, poses_array.shape[0]), axis=0)
        poses_array = poses_array.reshape((-1, self.window_size, poses_array.shape[-1]))
        trans_array = np.delete(trans_array, range(poses_array.shape[0] - remainder, poses_array.shape[0]), axis=0)
        trans_array = trans_array.reshape((-1, self.window_size, trans_array.shape[-1]))
        trans_vel_array = np.delete(trans_vel_array, range(poses_array.shape[0] - remainder, poses_array.shape[0]),
                                    axis=0)
        trans_vel_array = trans_vel_array.reshape((-1, self.window_size, trans_vel_array.shape[-1]))
        return poses_array, trans_array, trans_vel_array

    # Read names of sequence files from a txt
    def read_txt(self):
        with open(self.txt, "r") as f:
            self.sequences = [
                os.path.join(self._pose_path, line.replace("\n", "").split("_")[0],
                             line.replace("\n", "") + "_poses.npz")
                for line in f.readlines()
            ]

    def __len__(self):
        # Return the total number of batches
        return self._n_samples

    def __getitem__(self, idx):
        # Return the data for one batch
        poses = self._poses[idx]
        trans = self._trans[idx]
        vel = self._trans_vel[idx]
        beta = generate_continue_shape_param(seq_len=self.window_size)

        return poses, beta, vel, trans

    def on_epoch_end(self):
        if self.mode == "train":
            indices = np.arange(self._n_samples)
            random.shuffle(indices)
            self._poses = self._poses[indices]
            self._trans = self._trans[indices]
            self._trans_vel = self._trans_vel[indices]

    def __call__(self):
        for i in range(self.__len__()):
            yield self.__getitem__(i)

            if i == self.__len__() - 1:
                self.on_epoch_end()


def generate_continue_shape_param(seq_len=3):
    """
        Sample shape parameter (beta) from uniform distribution.
    """
    random_betas = tf.random.uniform(shape=(4,), minval=-2, maxval=2, dtype=tf.float32)
    remaining_zeros = tf.zeros(shape=(6,), dtype=tf.float32)
    betas = tf.concat([random_betas, remaining_zeros], axis=-1)
    return tf.tile(tf.expand_dims(betas, 0), [seq_len, 1])
