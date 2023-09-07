import argparse
import time
import sys
sys.path.append(".")

import tensorflow as tf
import script.utils.io as utils
from script.losses.utils import shape_id_beta
from script.train.model import DeformationModel
from script.utils.configparser import config_parser
from script.utils.global_vars import *


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default='config/predict.ini')
    opts = parser.parse_args()
    args_all = config_parser(os.path.join(ROOT_DIR, opts.config))
    args_eval = args_all['DEFAULT']

    gpus = tf.config.experimental.list_physical_devices("GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    if not gpus:
        print("No GPU detected")

    model_path = os.path.join(ROOT_DIR, args_eval['train_save_dir'], args_eval['eval_epoch'])

    validation_seq_list = args_eval['eval_motion'].replace(' ', '').split(',')
    shape_id_list = [s.strip() for s in args_eval['shape_id'].split(',')]
    print('model:', model_path)
    gaps = DeformationModel(args_eval)
    gaps.load_weights(model_path)

    fix_collision = args_eval.getboolean('fix_collisions')
    output_obj = args_eval.getboolean('output_obj')

    for motion in validation_seq_list:
        for shape_id in shape_id_list:
            print('motion:', motion, 'shape:', shape_id)
            motion_path = os.path.join(ROOT_DIR, args_eval['data_dir'], 'CMU', motion.split('_')[0],
                                       motion + '_poses.npz')

            # Load motion
            poses, trans, trans_vel = utils.load_motion(motion_path, window_size=1)
            num_frame = poses.shape[0]
            # Body shape
            betas = shape_id_beta()[int(shape_id)]
            betas = tf.repeat(tf.expand_dims(betas, 0), num_frame, axis=0)
            dir_name = 'with_postprocess' if fix_collision else 'no_postprocess'

            # Eval
            output_dir = os.path.join(ROOT_DIR, args_eval['eval_savedir'], args_eval['eval_epoch'],
                                      'shape' + shape_id, motion, dir_name)

            print('start predicting...')
            start_time = time.time()
            vg, vb = gaps.predict([poses, betas, trans_vel, trans],
                                      output_dir=output_dir,
                                      output_obj=output_obj,
                                      fix_collision=fix_collision)
            end_time = time.time()
            print('end.')

            print("Num frames: {}".format(num_frame))
            print("Total time(s): {}".format(end_time - start_time))
            print("Per frame time(s): {}".format((end_time - start_time) / num_frame))
            print("FPS: {}".format(num_frame / (end_time - start_time)))
