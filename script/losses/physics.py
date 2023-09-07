from script.losses.utils import *
from script.losses.layers import *


def deformation_gradient(triangles, Dm_inv):
    Ds = get_shape_matrix(triangles)
    return Ds @ Dm_inv 


def green_strain_tensor(F):
    I = tf.eye(2, dtype=F.dtype)
    Ft = tf.transpose(F, perm=[0, 1, 3, 2])
    return 0.5*(Ft @ F - I)
