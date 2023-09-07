from matplotlib import pyplot as plt

from script.losses.layers import NearestNeighbour
from script.losses.utils import *
from script.utils.io import get_garment_path, load_obj
from scipy.spatial.transform import Rotation as R
from script.train.smpl import VertexNormals
from script.utils.global_vars import ROOT_DIR
import os
root = os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

class Cloth():
    """
    Stores mesh and material information of the garment
    """
    def __init__(self, args_all, material, data_type=tf.float32):
        super(Cloth, self).__init__()
        self.data_type = tf.float32
        self.material = material
        self.args_expt = args_all
        print('garment:', self.args_expt['garment'])

        path = get_garment_path(self.args_expt['garment'])
        path = os.path.join(ROOT_DIR, path)

        if self.has_material_space:
            v, f, vm, fm = load_obj(path, tex_coords=True)
            self.em = get_vertex_connectivity(fm)
            vm = tf.convert_to_tensor(vm, data_type)
            fm = tf.convert_to_tensor(fm, tf.int32)
            self.vm = vm
            self.fm = fm
        else:
            v, f = load_obj(path, tex_coords=False)
        v = tf.convert_to_tensor(v, data_type)
        f = tf.convert_to_tensor(f, tf.int32)

        self.normals = face_normals(v, f).numpy()
        # Vertex attributes
        self.v_template = v
        self.v_mass = get_vertex_mass(v, f, self.material.density, data_type)
        self.v_velocity = tf.zeros((1, v.shape[0], 3), data_type)   # Vertex velocities in global coordinates
        # self.v = tf.zeros((1, v.shape[0], 3), data_type)          # Vertex position in global coordinates
        self.num_vertices = self.v_template.shape[0]
        self.v_weights = tf.zeros((self.num_vertices, 24))          # Vertex skinning weights

        # Face attributes
        self.f = f
        self.f_connectivity = get_face_connectivity(f)              # Pairs of adjacent faces
        self.f_connectivity_edges = get_face_connectivity_edges(f)  # Edges that connect faces
        self.f_area = tf.convert_to_tensor(get_face_areas(v, f), self.data_type)
        self.num_faces = self.f.shape[0]

        # Edge attributes
        self.e = get_vertex_connectivity(f)                         # Pairs of connected vertices
        self.e_rest = get_edge_length(v, self.e)                    # Rest length of the edges (world space)
        self.num_edges = self.e.shape[0]

        print('num vertices:', self.num_vertices)
        print('num faces:', self.num_faces)

        if 'pin_vertices' in self.args_expt:
            self.pin_vertices = np.fromstring(self.args_expt['pin_vertices'], sep=',')
            self.pin_vertices = tf.convert_to_tensor(self.pin_vertices, dtype=tf.int32)

        if self.has_material_space:
            # Rest state of the cloth (computed in material space)
            tri_m = gather_triangles(vm, fm)
            self.Dm = get_shape_matrix(tri_m)
            self.Dm_inv = tf.linalg.inv(self.Dm)

            self.em = get_vertex_connectivity(fm)
            self.em_rest = get_edge_length(vm, self.em)
            self.em_dict = get_v_connect_dict(self.em)
        else:
            self.Dm_inv = self.make_continuum()
            self.Dm_inv = tf.convert_to_tensor(self.Dm_inv, dtype=data_type)

        self.e_dict = get_v_connect_dict(self.e)
        self.ragged_neigh = dict_to_ragged_tensor(self.e_dict)
        self.isometry_covar = isometry_covar(tf.expand_dims(self.v_template, axis=0), self.ragged_neigh)
        self.eig_val, _ = tf.linalg.eigh(self.isometry_covar)   # 3 Eigenvalues

    @property
    def has_material_space(self):
        # if the original obj file provide (2D) material space.
        # For t-shirt, tank and long-sleeve dress, there is a non-negligible discrepancy between 3D space
        # and material space, so we discard them.
        return self.args_expt['garment'] in ['pants', 'shorts', 'dress']

    def compute_closest_skinning_weights(self, smpl):
        self.closest_body_vertices = find_nearest_neighbour(self.v_template, smpl.template_vertices)
        self.v_weights = tf.gather(smpl.skinning_weights, self.closest_body_vertices).numpy()
        self.v_weights = tf.convert_to_tensor(self.v_weights, dtype=self.data_type)

    def compute_k_nearest_skinning_weights(self, smpl, k=30):
        # squared distance
        garment_body_distance = pairwise_distance(self.v_template, smpl.template_vertices)

        closest_indices_body = tf.argsort(garment_body_distance, axis=1)
        contribution = tf.zeros([self.num_vertices, smpl.num_vertices])
        B = closest_indices_body[:, :k]
        updates = tf.ones_like(B, dtype=contribution.dtype) * 1 / k

        rows = tf.range(B.shape[0])[:, tf.newaxis]
        rows_repeated = tf.repeat(rows, B.shape[1], axis=1)
        indices = tf.stack([rows_repeated, B], axis=-1)
        contribution = tf.tensor_scatter_nd_update(contribution, indices, updates)

        w_skinning = tf.matmul(contribution, smpl.skinning_weights)
        self.v_weights = tf.convert_to_tensor(w_skinning, dtype=self.data_type)

    def compute_rbf_skinning_weight(self, smpl):
        def gaussian_rbf(distances, min_dist):
            """
            Compute weights using a Gaussian RBF.
            Returns:
                 [num_vert_garment, num_vert_body]
            """

            # Compute the weights using the Gaussian RBF formula
            sigma = min_dist + 1e-7
            k = 0.25

            weights = tf.exp(
                -(distances - tf.expand_dims(min_dist, axis=1)) ** 2 / tf.expand_dims(k * sigma ** 2, axis=1))
            # Normalize the weights
            weights = weights / tf.reduce_sum(weights, axis=1, keepdims=True)
            return weights

        # distance
        garment_body_distance = tf.sqrt(pairwise_distance(self.v_template, smpl.template_vertices))
        min_dist = tf.reduce_min(garment_body_distance, axis=1)  # (n_vert_garment, )
        weight = gaussian_rbf(garment_body_distance, min_dist)

        w_skinning = tf.matmul(weight, smpl.skinning_weights)

        self.v_weights = tf.convert_to_tensor(w_skinning, dtype=self.data_type)

    def compute_softmax_skinning_weight(self, smpl):
        # distance
        garment_body_distance = tf.sqrt(pairwise_distance(self.v_template, smpl.template_vertices))
        weight = tf.nn.softmax(1 / garment_body_distance)
        w_skinning = tf.matmul(weight, smpl.skinning_weights)
        self.v_weights = tf.convert_to_tensor(w_skinning, dtype=self.data_type)

    def compute_bending_coeff(self, smpl):
        va = self.v_template[None, :]
        vb = smpl.template_vertices[None, :]
        nb = VertexNormals()(vb, smpl.faces)
        closest_vertices = NearestNeighbour()(va, vb)

        vb = tf.gather(vb, closest_vertices, batch_dims=1)
        nb = tf.gather(nb, closest_vertices, batch_dims=1)

        distance = tf.reduce_sum(nb * (va - vb), axis=-1)

        dist_a = tf.gather(distance, self.f_connectivity_edges[:, 0], axis=1)
        dist_b = tf.gather(distance, self.f_connectivity_edges[:, 1], axis=1)
        dist_edge_body = 0.5 * (dist_a + dist_b)
        min_dist = tf.reduce_min(dist_edge_body)
        max_dist = tf.reduce_max(dist_edge_body)
        w_bend_template = (dist_edge_body - min_dist) / (max_dist - min_dist)
        return w_bend_template

    def make_continuum(self):
        """
        Calculate material space uv
        """
        f = self.f.numpy()
        angle = np.arccos(self.normals[:, 2])
        axis = np.stack(
            [
                self.normals[:, 1],
                -self.normals[:, 0],
                np.zeros((self.normals.shape[0],), np.float32),
            ],
            axis=-1,
        )
        axis /= np.linalg.norm(axis, axis=-1, keepdims=True)
        axis_angle = axis * angle[..., None]
        rotations = R.from_rotvec(axis_angle).as_matrix()
        triangles = self.v_template.numpy()[f]
        triangles = np.einsum("abc,adc->abd", triangles, rotations)

        triangles = triangles[..., :2]
        uv_matrices = np.stack(
            [triangles[:, 0] - triangles[:, 2], triangles[:, 1] - triangles[:, 2]],
            axis=-1,
        )
        Dm_inv = np.linalg.inv(uv_matrices)
        return Dm_inv

    @property
    def pinning(self):
        return hasattr(self, "pin_vertices")

    def map_indices_to_3D_space(self, em_dict, mapping):
        e_dict = {}
        seam_idx = []
        seam_idx_m = []
        mapping_inv = {}        # k to km mapping
        for k_m, v_m in em_dict.items():
            k = mapping[k_m]
            v = [mapping[vv] for vv in v_m]
            if k in e_dict:
                seam_idx_m.append(k_m)
                if k not in seam_idx:
                    seam_idx.append(k)
                if mapping_inv[k] not in seam_idx_m:
                    seam_idx_m.append(mapping_inv[k])
            else:
                e_dict[k] = v
                mapping_inv[k] = k_m
        for d in seam_idx:
            del e_dict[d]
        for d in seam_idx_m:
            del em_dict[d]
        return e_dict, em_dict


@tf.function
def face_normals(vertices, faces, normalized=True):
    input_shape = vertices.get_shape()
    vertices = tf.reshape(vertices, (-1, *input_shape[-2:]))
    v01 = tf.gather(vertices, faces[:, 1], axis=1) - tf.gather(
        vertices, faces[:, 0], axis=1
    )
    v12 = tf.gather(vertices, faces[:, 2], axis=1) - tf.gather(
        vertices, faces[:, 1], axis=1
    )
    normals = tf.linalg.cross(v01, v12)
    if normalized:
        normals /= tf.norm(normals, axis=-1, keepdims=True) + tf.keras.backend.epsilon()
    normals = tf.reshape(normals, (*input_shape[:-2], -1, 3))
    return normals


