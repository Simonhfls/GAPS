import numpy as np
import tensorflow as tf


def pairwise_distance(A, B):
    rA = np.sum(np.square(A), axis=1)
    rB = np.sum(np.square(B), axis=1)
    distances = - 2 * np.matmul(A, np.transpose(B)) + rA[:, np.newaxis] + rB[np.newaxis, :]
    return distances


def find_nearest_neighbour(A, B, dtype=np.int32):
    nearest_neighbour = np.argmin(pairwise_distance(A, B), axis=1)
    return nearest_neighbour.astype(dtype)


def get_vertex_connectivity(faces, dtype=tf.int32):
    """
    Returns a list of unique edges in the mesh.
    Each edge contains the indices of the vertices it connects
    """
    if tf.is_tensor(faces):
        faces = faces.numpy()

    edges = set()
    for f in faces:
        num_vertices = len(f)
        for i in range(num_vertices):
            j = (i + 1) % num_vertices
            edges.add(tuple(sorted([f[i], f[j]])))

    return tf.convert_to_tensor(list(edges), dtype)


def get_face_connectivity(faces, dtype=tf.int32):
    """
    Returns a list of adjacent face pairs
    """
    if tf.is_tensor(faces):
        faces = faces.numpy()

    edges = get_vertex_connectivity(faces).numpy()

    G = {tuple(e): [] for e in edges}
    for i, f in enumerate(faces):
        n = len(f)
        for j in range(n):
            k = (j + 1) % n
            e = tuple(sorted([f[j], f[k]]))
            G[e] += [i]

    adjacent_faces = []
    for key in G:
        assert len(G[key]) < 3
        if len(G[key]) == 2:
            adjacent_faces += [G[key]]

    return tf.convert_to_tensor(adjacent_faces, dtype)


def get_face_connectivity_edges(faces, dtype=tf.int32):
    """
    Returns a list of edges that connect two faces
    (i.e., all the edges except borders)
    """
    if tf.is_tensor(faces):
        faces = faces.numpy()

    edges = get_vertex_connectivity(faces).numpy()

    G = {tuple(e): [] for e in edges}
    for i, f in enumerate(faces):
        n = len(f)
        for j in range(n):
            k = (j + 1) % n
            e = tuple(sorted([f[j], f[k]]))
            G[e] += [i]

    adjacent_face_edges = []
    for key in G:
        assert len(G[key]) < 3
        if len(G[key]) == 2:
            adjacent_face_edges += [list(key)]

    return tf.convert_to_tensor(adjacent_face_edges, dtype)


def get_vertex_mass(vertices, faces, density, dtype=tf.float32):
    """
        Computes the mass of each vertex according to triangle areas and fabric density
    """
    areas = get_face_areas(vertices, faces)
    triangle_masses = density * areas

    vertex_masses = np.zeros(vertices.shape[0])
    np.add.at(vertex_masses, faces[:, 0], triangle_masses / 3)
    np.add.at(vertex_masses, faces[:, 1], triangle_masses / 3)
    np.add.at(vertex_masses, faces[:, 2], triangle_masses / 3)

    return tf.convert_to_tensor(vertex_masses, dtype)


def get_face_areas(vertices, faces):
    if tf.is_tensor(vertices):
        vertices = vertices.numpy()

    if tf.is_tensor(faces):
        faces = faces.numpy()

    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]

    u = v2 - v0
    v = v1 - v0

    return np.linalg.norm(np.cross(u, v), axis=-1) / 2.0


def get_face_areas_batch(vertices, faces):
    v0 = tf.gather(vertices, faces[:, 0], axis=1)
    v1 = tf.gather(vertices, faces[:, 1], axis=1)
    v2 = tf.gather(vertices, faces[:, 2], axis=1)

    u = v2 - v0
    v = v1 - v0

    return tf.norm(tf.linalg.cross(u, v), axis=-1) / 2.0


def get_edge_length(vertices, edges):
    v0 = tf.gather(vertices, edges[:, 0], axis=-2)
    v1 = tf.gather(vertices, edges[:, 1], axis=-2)
    return tf.linalg.norm(v0 - v1, axis=-1)


def get_shape_matrix(x):
    if x.shape.ndims == 3:
        return tf.stack([x[:, 0] - x[:, 2], x[:, 1] - x[:, 2]], axis=-1)

    elif x.shape.ndims == 4:
        return tf.stack([x[:, :, 0] - x[:, :, 2], x[:, :, 1] - x[:, :, 2]], axis=-1)

    raise NotImplementedError


def gather_triangles(vertices, indices):
    if vertices.shape.ndims == (indices.shape.ndims + 1):
        indices = tf.repeat([indices], tf.shape(vertices)[0], axis=0)

    triangles = tf.gather(vertices, indices,
                          axis=-2,
                          batch_dims=vertices.shape.ndims - 2)

    return triangles


def fix_collisions(vc, vb, nb, eps=0.002):
    """
    Fix the collisions between the clothing and the body by projecting
    the clothing's vertices outside the body's surface
    """

    # For each vertex of the cloth, find the closest vertices in the body's surface
    closest_vertices = find_nearest_neighbour(vc, vb)
    vb = vb[closest_vertices]
    nb = nb[closest_vertices]

    # Test penetrations
    penetrations = np.sum(nb * (vc - vb), axis=1) - eps
    penetrations = np.minimum(penetrations, 0)

    # Fix the clothing
    corrective_offset = -np.multiply(nb, penetrations[:, np.newaxis])
    vc_fixed = vc + corrective_offset
    return vc_fixed


def get_v_connect_dict(vertex_connectivity):
    vertex_connectivity = vertex_connectivity.numpy()
    vertex_dict = {}

    # Iterate over each row in the vertex connectivity tensor
    for i in range(vertex_connectivity.shape[0]):
        vertex1, vertex2 = vertex_connectivity[i]

        # Add vertex2 to the connected vertex IDs of vertex1
        if vertex1 in vertex_dict:
            vertex_dict[vertex1].append(vertex2)
        else:
            vertex_dict[vertex1] = [vertex2]

        # Add vertex1 to the connected vertex IDs of vertex2
        if vertex2 in vertex_dict:
            vertex_dict[vertex2].append(vertex1)
        else:
            vertex_dict[vertex2] = [vertex1]
    return vertex_dict


def dict_to_ragged_tensor(dic):
    values = []
    row_splits = [0]
    for key in sorted(dic.keys()):
        values.extend(dic[key])
        row_splits.append(row_splits[-1] + len(dic[key]))
    ragged = tf.RaggedTensor.from_row_splits(values, row_splits)
    return ragged


def isometry_covar(v, ragged):
    """
     compute covariance matrix.
        1/K * (X_mean-Xn)'@(X_mean-Xn)
    """
    neigh = tf.gather(v, ragged, axis=1)
    diff = (neigh - tf.reduce_mean(neigh, axis=-2, keepdims=True)).to_tensor()
    num_neigh = tf.cast(neigh.row_lengths(axis=2), diff.dtype).to_tensor()
    covar = tf.matmul(diff, diff, transpose_a=True) / tf.expand_dims(tf.expand_dims(num_neigh, -1), -1)
    return covar


def matrix_determinant(M):
    """
    input:
    - M: [48, 4424, 3, 3, 3]
    output:
        determinant of shape [48, 4424, 3, 3]
    """

    det = M[..., 0, 0] * (M[..., 1, 1] * M[..., 2, 2] - M[..., 2, 1] * M[..., 1, 2]) - \
          M[..., 0, 1] * (M[..., 1, 0] * M[..., 2, 2] - M[..., 1, 2] * M[..., 2, 0]) + \
          M[..., 0, 2] * (M[..., 1, 0] * M[..., 2, 1] - M[..., 2, 0] * M[..., 1, 1])

    return det


def shape_id_beta():
    betas = tf.constant([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                          [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                          [0, -1, 0, 0, 0, 0, 0, 0, 0, 0],
                          [0, 2, 0, 0, 0, 0, 0, 0, 0, 0],
                          [0, -2, 0, 0, 0, 0, 0, 0, 0, 0],
                          [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                          [0, 0, -1, 0, 0, 0, 0, 0, 0, 0],
                          [0, 0, 2, 0, 0, 0, 0, 0, 0, 0],
                          [0, 0, -2, 0, 0, 0, 0, 0, 0, 0],
                          [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                          [-1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                          [2, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                          [-2, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                          [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                          [0, 0, 0, -1, 0, 0, 0, 0, 0, 0],
                          [0, 0, 0, 2, 0, 0, 0, 0, 0, 0],
                          [0, 0, 0, -2, 0, 0, 0, 0, 0, 0],
                         ], dtype=tf.float32)

    return betas