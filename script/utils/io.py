import os
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
from scipy.spatial.transform import Rotation as R


def load_obj(filename, tex_coords=False):
    vertices = []
    faces = []
    uvs = []
    faces_uv = []

    with open(filename, 'r') as fp:
        for line in fp:
            line_split = line.split()

            if not line_split:
                continue

            elif tex_coords and line_split[0] == 'vt':
                uvs.append([line_split[1], line_split[2]])

            elif line_split[0] == 'v':
                vertices.append([line_split[1], line_split[2], line_split[3]])

            elif line_split[0] == 'f':
                vertex_indices = [s.split("/")[0] for s in line_split[1:]]
                faces.append(vertex_indices)

                if tex_coords:
                    uv_indices = [s.split("/")[1] for s in line_split[1:]]
                    faces_uv.append(uv_indices)

    vertices = np.array(vertices, dtype=np.float32)
    faces = np.array(faces, dtype=np.int32) - 1

    if tex_coords:
        uvs = np.array(uvs, dtype=np.float32)
        faces_uv = np.array(faces_uv, dtype=np.int32) - 1
        return vertices, faces, uvs, faces_uv

    return vertices, faces


def save_obj(filename, vertices, faces, rgb=None):
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    if tf.is_tensor(vertices):
        vertices = vertices.numpy()
    
    if tf.is_tensor(faces):
        faces = faces.numpy()

    vertices = vertices.squeeze()
    faces = faces.squeeze()

    if rgb:
        r, g, b = rgb_to_float(*rgb)

    with open(filename, 'w') as fp:
        if not rgb:
            for v in vertices:
                fp.write('v %f %f %f\n' % (v[0], v[1], v[2]))
        else:
            for v in vertices:
                fp.write('v %f %f %f' % (v[0], v[1], v[2]))
                fp.write(' %f %f %f\n' % (r, g, b))

        for f in (faces + 1):  # Faces are 1-based, not 0-based in obj files
            fp.write('f %d %d %d\n' % (f[0], f[1], f[2]))
    
    print("Saved:", filename)

def save_obj_with_isometry_color(filename, vertices, faces, isometry_measure, colormap=False):
    def values_to_RdBu(values):
        colormap = plt.get_cmap("RdBu_r")
        normalized_values = (np.array(values) + 1) / 2  # Normalize values from [-1, 1] to [0, 1]
        colors = colormap(normalized_values)
        return colors[:, :3]  # Exclude the alpha channel
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    # mask = tf.math.logical_and(isometry_measure > -0.01, isometry_measure < 0.01)
    # mask = isometry_measure > 0.2
    # isometry_measure = tf.where(mask, 0.2, isometry_measure)

    if tf.is_tensor(vertices):
        vertices = vertices.numpy()

    if tf.is_tensor(faces):
        faces = faces.numpy()

    vertices = vertices.squeeze()
    faces = faces.squeeze()

    if not colormap:
        with open(filename, 'w') as fp:
            for v, alpha in zip(vertices, isometry_measure):
                fp.write('v %f %f %f' % (v[0], v[1], v[2]))
                fp.write(' %f %f %f' % (1, 0, 0))
                fp.write(' %f\n' % alpha)

            for f in (faces + 1):  # Faces are 1-based, not 0-based in obj files
                fp.write('f %d %d %d\n' % (f[0], f[1], f[2]))

    else:
        RdBu = values_to_RdBu(isometry_measure)
        with open(filename, 'w') as fp:
            for v, rgb in zip(vertices, RdBu):
                fp.write('v %f %f %f' % (v[0], v[1], v[2]))
                fp.write(' %f %f %f\n' % (rgb[0], rgb[1], rgb[2]))

            for f in (faces + 1):  # Faces are 1-based, not 0-based in obj files
                fp.write('f %d %d %d\n' % (f[0], f[1], f[2]))
    print("Saved:", filename)


def rgb_to_float(r, g, b):
    return r / 255.0, g / 255.0, b / 255.0


def load_motion(path, window_size=3):
    motion = np.load(path, mmap_mode='r')

    reduce_factor = int(motion['mocap_framerate'] // 30)
    pose = motion['poses'][::reduce_factor, :72]
    trans = motion['trans'][::reduce_factor, :]

    separate_arms(pose)

    # Swap axes
    swap_rotation = R.from_euler('zx', [-90, 270], degrees=True)
    root_rot = R.from_rotvec(pose[:, :3])
    pose[:, :3] = (swap_rotation * root_rot).as_rotvec()
    trans = swap_rotation.apply(trans)

    # Center model in first frame
    trans = trans - trans[0]

    # Compute velocities
    trans_vel = finite_diff(trans, 1 / 30)

    # Make sure the sequence is of length window_size * N
    remainder = pose.shape[0] % window_size
    pose = np.delete(pose, range(pose.shape[0] - remainder, pose.shape[0]), axis=0)
    trans = np.delete(trans, range(trans.shape[0] - remainder, trans.shape[0]), axis=0)
    trans_vel = np.delete(trans_vel, range(trans_vel.shape[0] - remainder, trans_vel.shape[0]), axis=0)

    return pose.astype(np.float32), trans.astype(np.float32), trans_vel.astype(np.float32)


def separate_arms(poses, angle=20, left_arm=17, right_arm=16):
    num_joints = poses.shape[-1] // 3

    poses = poses.reshape((-1, num_joints, 3))
    rot = R.from_euler('z', -angle, degrees=True)
    poses[:, left_arm] = (rot * R.from_rotvec(poses[:, left_arm])).as_rotvec()
    rot = R.from_euler('z', angle, degrees=True)
    poses[:, right_arm] = (rot * R.from_rotvec(poses[:, right_arm])).as_rotvec()

    poses[:, 23] *= 0.1
    poses[:, 22] *= 0.1

    return poses.reshape((poses.shape[0], -1))


def finite_diff(x, h, diff=1):
    if diff == 0:
        return x

    v = np.zeros(x.shape, dtype=x.dtype)
    v[1:] = (x[1:] - x[0:-1]) / h

    return finite_diff(v, h, diff-1)


def check_gpus():
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    if tf.test.gpu_device_name():
        print('**** Default GPU Device:{} ****'.format(tf.test.gpu_device_name()))
    else:
        print("**** Please install GPU version of TF **** ")


def get_garment_path(garment):
    garments = {
        "tshirt": "assets/meshes/tshirt.obj",
        "tank": "assets/meshes/tank.obj",
        "top": "assets/meshes/long_sleeve_top.obj",
        "pants": "assets/meshes/pants.obj",
        "shorts": "assets/meshes/shorts.obj",
        "dress": "assets/meshes/dress.obj",
        "skirt": "assets/meshes/straight_skirt.obj",
        "umb_skirt": "assets/meshes/umbrella_skirt.obj",
        "long_sleeve_dress": "assets/meshes/long_sleeve_dress.obj",
        "sleeveless_dress": "assets/meshes/sleeveless_dress.obj",
    }

    assert garment in garments, f"'{garment}' is not a valid option. Valid options: {list(garments.keys())}"
    return garments[garment]
