import math
import numpy as np
import tensorflow as tf
from script.losses.layers import NearestNeighbour, FaceNormals
from script.losses.physics import green_strain_tensor, deformation_gradient
from script.losses.utils import get_edge_length, gather_triangles, isometry_covar, matrix_determinant, \
     get_face_areas_batch


# Mass-spring model
class EdgeLoss:
    def __init__(self, template):
        self.el_template = template.e_rest
        self.e = template.e
        self.f_area = template.f_area
        self.f = template.f

    @tf.function
    def __call__(self, v):
        batch_size = tf.cast(tf.shape(v)[0], v.dtype)
        el_garment = get_edge_length(v, self.e)
        edge_difference = el_garment - self.el_template

        loss = tf.reduce_sum(edge_difference ** 2)
        loss = loss / batch_size

        diff_el_pct = tf.abs(edge_difference) / self.el_template
        metric = tf.reduce_mean(diff_el_pct)  # edge elongation in %

        area_garment = get_face_areas_batch(v, self.f)
        diff_area = tf.abs(area_garment - self.f_area)
        diff_area_pct = diff_area / self.f_area
        metric_area = tf.reduce_mean(diff_area_pct)

        return loss, metric, metric_area


# Saint-Venant Kirchhoff
class StVKLoss:
    def __init__(self, template):
        self.template = template

    @tf.function
    def __call__(self, v):
        batch_size = tf.cast(tf.shape(v)[0], v.dtype)
        triangles = gather_triangles(v, self.template.f)
        Dm_inv = tf.repeat([self.template.Dm_inv], tf.shape(v)[0], axis=0)

        F = deformation_gradient(triangles, Dm_inv)
        G = green_strain_tensor(F)

        # Energy
        mat = self.template.material
        I = tf.eye(2, batch_shape=tf.shape(G)[:2], dtype=G.dtype)
        S = mat.lame_mu * G + 0.5 * mat.lame_lambda * tf.linalg.trace(G)[:, :, tf.newaxis, tf.newaxis] * I
        energy_density = tf.linalg.trace(tf.transpose(S, [0, 1, 3, 2]) @ G)
        energy = self.template.f_area[tf.newaxis] * mat.thickness * energy_density
        loss = tf.reduce_sum(energy) / batch_size

        el_template = self.template.e_rest
        el_garment = get_edge_length(v, self.template.e)
        diff_el = tf.abs(el_garment - el_template)
        diff_el_pct = diff_el / el_template
        metric_edge = tf.reduce_mean(diff_el_pct)       # edge difference in %

        area_garment = get_face_areas_batch(v, self.template.f)
        diff_area = tf.abs(area_garment - self.template.f_area)
        diff_area_pct = diff_area / self.template.f_area
        metric_area = tf.reduce_mean(diff_area_pct)     # area difference in %

        return loss, metric_edge, metric_area


class BendingLoss:
    def __init__(self, template, follow_template_weight=None):
        self.template = template
        v_template = self.template.v_template[np.newaxis, :]
        # Compute face normals
        fn = FaceNormals(dtype=v_template.dtype)(v_template, self.template.f)

        n0 = tf.gather(fn, self.template.f_connectivity[:, 0], axis=1)
        n1 = tf.gather(fn, self.template.f_connectivity[:, 1], axis=1)

        # Compute edge length
        v0 = tf.gather(v_template, self.template.f_connectivity_edges[:, 0], axis=1)
        v1 = tf.gather(v_template, self.template.f_connectivity_edges[:, 1], axis=1)
        e = v1 - v0
        e_norm, l = tf.linalg.normalize(e, axis=-1)

        # Compute template dihedral angle between faces
        cos = tf.reduce_sum(tf.multiply(n0, n1), axis=-1)
        sin = tf.reduce_sum(tf.multiply(e_norm, tf.linalg.cross(n0, n1)), axis=-1)
        self.theta_template = tf.math.atan2(sin, cos)
        self.follow_template_weight = follow_template_weight

    @tf.function
    def __call__(self, v):
        batch_size = tf.cast(tf.shape(v)[0], v.dtype)

        # Compute face normals
        fn = FaceNormals(dtype=v.dtype)(v, self.template.f)
        n0 = tf.gather(fn, self.template.f_connectivity[:, 0], axis=1)
        n1 = tf.gather(fn, self.template.f_connectivity[:, 1], axis=1)

        # Compute edge length
        v0 = tf.gather(v, self.template.f_connectivity_edges[:, 0], axis=1)
        v1 = tf.gather(v, self.template.f_connectivity_edges[:, 1], axis=1)
        e = v1 - v0
        e_norm, l = tf.linalg.normalize(e, axis=-1)

        # Compute area
        f_area = tf.repeat([self.template.f_area], tf.shape(v)[0], axis=0)
        a0 = tf.gather(f_area, self.template.f_connectivity[:, 0], axis=1)
        a1 = tf.gather(f_area, self.template.f_connectivity[:, 1], axis=1)
        a = a0 + a1

        # Compute dihedral angle between faces
        cos = tf.reduce_sum(tf.multiply(n0, n1), axis=-1)
        sin = tf.reduce_sum(tf.multiply(e_norm, tf.linalg.cross(n0, n1)), axis=-1)
        theta = tf.math.atan2(sin, cos)

        mat = self.template.material
        scale = l[..., 0] ** 2 / (4 * a)

        # Bending energy
        energy1 = mat.bending_coeff * scale * (theta ** 2) / 2
        energy2 = mat.bending_coeff * scale * ((self.theta_template - theta) ** 2) / 2
        if self.follow_template_weight is not None:
            energy = (1 - self.follow_template_weight) * energy1 + self.follow_template_weight * energy2
        else:
            energy = 0.5 * energy1 + 0.5 * energy2

        loss = tf.reduce_sum(energy) / batch_size
        metric = tf.reduce_mean(tf.abs(theta)) * 180 / math.pi
        return loss, metric


class CollisionLoss:
    def __init__(self, eps=2e-3, return_average=True):
        self.eps = eps
        self.return_average = return_average

    @tf.function
    def __call__(self, va, vb, nb):
        batch_size = tf.cast(tf.shape(va)[0], va.dtype)
        bs_int = tf.shape(va)[0]
        num_frames = tf.shape(va)[1]
        loss = 0.0
        vb = tf.reshape(vb, (batch_size, num_frames, -1, 3))
        nb = tf.reshape(nb, (batch_size, num_frames, -1, 3))
        distances = tf.TensorArray(tf.float32, size=num_frames)
        for i in tf.range(num_frames):
            va_frame = va[:, i]
            vb_frame = vb[:, i]
            nb_frame = nb[:, i]
            closest_vertices = NearestNeighbour(dtype=va.dtype)(va_frame, vb_frame)
            vb_frame = tf.gather(vb_frame, closest_vertices, batch_dims=1)
            nb_frame = tf.gather(nb_frame, closest_vertices, batch_dims=1)
            distance = tf.reduce_sum(nb_frame * (va_frame - vb_frame), axis=-1)
            distances = distances.write(i, distance)
            interpenetration = tf.maximum(self.eps - distance, 0)
            loss += tf.reduce_sum(interpenetration ** 3)

        distances = distances.stack()
        distances = tf.reshape(tf.transpose(distances, perm=[1, 0, 2]), (bs_int * num_frames, -1))
        # percentage of collision
        metric = tf.reduce_mean(tf.cast(tf.less(distances, 0), tf.float32))

        if self.return_average:
            loss = loss / batch_size

        return loss, metric, distances


class GravityLoss:
    def __init__(self, mass, gravity=9.81):
        self.mass = mass
        self.g = gravity

    @tf.function
    def __call__(self, x):
        batch_size = tf.cast(tf.shape(x)[0], x.dtype)
        U = self.g * self.mass[tf.newaxis, tf.newaxis] * x[:, :, 1]  # y-axis is pointing up
        loss = tf.reduce_sum(U) / batch_size
        return loss


class InertiaLoss:
    def __init__(self, mass, dt):
        self.vertex_mass = mass
        self.dt = dt

    @tf.function
    def __call__(self, vertices):
        x0 = vertices[:, 0]
        x1 = vertices[:, 1]
        x2 = vertices[:, 2]
        x_proj = 2 * x1 - x0
        x_proj = tf.stop_gradient(x_proj)
        dx = x2 - x_proj
        loss = (0.5 / self.dt**2) * self.vertex_mass[:, None] * dx**2
        loss = tf.reduce_mean(loss, axis=0)
        loss = tf.reduce_sum(loss)
        return loss


class IsometryLoss:
    def __init__(self, garment):
        self.template = garment

    @tf.function
    def __call__(self, vertices):
        batch_size = tf.cast(tf.shape(vertices)[0], vertices.dtype)
        covar_gar = isometry_covar(vertices, self.template.ragged_neigh)

        num_verts = tf.shape(covar_gar)[1]
        covar_gar = tf.stack([covar_gar, covar_gar, covar_gar], axis=-3)  # [48, 4424, 3, 3, 3]
        # [1, 4424, 3, 3, 3]
        lmb_I = tf.eye(3, batch_shape=[1, num_verts, 3]) * tf.expand_dims(tf.expand_dims(self.template.eig_val, -1),-1)

        M = covar_gar - lmb_I
        det_M = matrix_determinant(M)

        loss = tf.reduce_sum(tf.abs(det_M))
        loss = loss / batch_size

        el_template = self.template.e_rest
        el_garment = get_edge_length(vertices, self.template.e)
        diff_el = abs(el_garment - el_template)
        diff_el_pct = diff_el / el_template
        metric = tf.reduce_mean(diff_el_pct)  # edge elongation in %
        return loss, metric


class CAIsometryLoss:
    """
    Collision aware isometry loss
    """
    def __init__(self, garment, k_amp=10):
        self.template = garment
        self.k_amp = k_amp

    @tf.function
    def __call__(self, vertices, distances, epoch):
        batch_size = tf.cast(tf.shape(vertices)[0], vertices.dtype)
        eps = 2e-3
        interpenetration = tf.maximum(eps - distances, 0)
        ext_coef = 1 + tf.minimum(interpenetration * self.k_amp, 0.03) * tf.minimum(epoch, 100)     # OK
        ext_coef = tf.expand_dims(tf.expand_dims(tf.expand_dims(ext_coef, -1), -1), -1)
        covar_gar = isometry_covar(vertices, self.template.ragged_neigh)

        num_verts = tf.shape(covar_gar)[1]

        covar_gar = tf.stack([covar_gar, covar_gar, covar_gar], axis=-3)  # [48, 4424, 3, 3, 3]
        # [1, 4424, 3, 3, 3]
        lmb_I = tf.eye(3, batch_shape=[1, num_verts, 3]) * tf.expand_dims(tf.expand_dims(self.template.eig_val, -1), -1)

        E_lmb_I = ext_coef * lmb_I
        M = covar_gar - E_lmb_I

        det_M = matrix_determinant(M)
        loss = tf.reduce_sum(tf.abs(det_M))
        loss = loss / batch_size

        el_template = self.template.e_rest
        el_garment = get_edge_length(vertices, self.template.e)
        diff_el = abs(el_garment - el_template)
        diff_el_pct = diff_el / el_template
        metric = tf.reduce_mean(diff_el_pct)    # edge elongation in %
        return loss, metric


class PinningLoss:
    def __init__(self, garment):
        self.indices = garment.pin_vertices
        self.vertices = tf.gather(garment.v_template, self.indices)

    @tf.function
    def __call__(self, unskinned):
        loss = tf.gather(unskinned, self.indices, axis=-2) - self.vertices
        loss = loss[..., 1] ** 2        # preserve y coordinates
        loss = tf.reduce_sum(loss, axis=-1)
        loss = tf.reduce_mean(loss)
        return loss