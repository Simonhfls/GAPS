import os
from tensorflow.keras.layers import GRU, Dense, RNN, GRUCell
from script.train import smpl
from script.losses.cloth import Cloth
from script.losses.losses import *
from script.losses.material import Material
from script.losses.metrics import MyMetric
from script.losses.utils import fix_collisions
from script.train.smpl import VertexNormals
from script.utils.global_vars import ROOT_DIR
import script.utils.io as utils


class DeformationModel(tf.keras.Model):
    def __init__(self, config):
        super(DeformationModel, self).__init__()
        self.config = config
        # Load smpl
        self.body = smpl.SMPL(os.path.join(ROOT_DIR, config['smpl_path']))

        # Fabric material parameters
        thickness = config.getfloat('thickness')        # (m)
        bulk_density = config.getfloat('bulk_density')  # (kg / m3)
        area_density = thickness * bulk_density
        young_modulus = config.getfloat('young_modulus')
        poisson_ratio = config.getfloat('poisson_ratio')

        material = Material(
            density=area_density,  # Fabric density (kg / m2)
            thickness=thickness,  # Fabric thickness (m)
            young_modulus=young_modulus,
            poisson_ratio=poisson_ratio,
            stretch_multiplier=config.getfloat('stretch_multiplier'),
            bending_multiplier=config.getfloat('bending_multiplier')
        )

        self.garment = Cloth(args_all=config, material=material)

        if self.config['mode'] == 'train':
            if 'skin_method' not in self.config or self.config['skin_method'] not in ('rbf', 'k_nearest', 'closest'):
                raise ValueError('Please specify a skin_method')
            if self.config['skin_method'] == 'rbf':
                print('compute rbf skinning weight...')
                self.garment.compute_rbf_skinning_weight(self.body)
            elif self.config['skin_method'] == 'k_nearest':
                print('compute k_nearest skinning weight...')
                self.garment.compute_k_nearest_skinning_weights(self.body)
            elif self.config['skin_method'] == 'closest':
                print('compute closest skinning weight...')
                self.garment.compute_closest_skinning_weights(self.body)

        self.w_bending_follow_template = self.garment.compute_bending_coeff(self.body)
        if 'collision_k_amp' not in self.config:
            self.col_k_amp = 10
        else:
            self.col_k_amp = self.config.getfloat('collision_k_amp')
        self.build_model()
        # Losses/Metrics
        self.build_losses_and_metrics()
        self.current_epoch = tf.Variable(0, trainable=False, dtype=tf.float32)

    def build_model(self):
        self.gru1 = RNN(GRUCell(256, activation='tanh'), return_sequences=True, return_state=True)
        self.gru2 = RNN(GRUCell(256, activation='tanh'), return_sequences=True, return_state=True)
        self.gru3 = RNN(GRUCell(256, activation='tanh'), return_sequences=True, return_state=True)
        self.gru4 = RNN(GRUCell(256, activation='tanh'), return_sequences=True, return_state=True)
        self.linear = Dense(self.garment.v_template.shape[0] * 3)
        self.skin_weight = tf.Variable(self.garment.v_weights,
                                       trainable=self.config.getboolean('opt_skinning_weight'),
                                       name='w_skin')

    def build_losses_and_metrics(self):
        # Losses and Metrics
        self.loss_metric = MyMetric(name="Loss")
        # Cloth model
        if self.config['cloth_model'] == "mass-spring":
            self.cloth_loss = EdgeLoss(self.garment)
        elif self.config['cloth_model'] == "stvk":
            self.cloth_loss = StVKLoss(self.garment)
        self.strain_metric = MyMetric(name="M_Strain")
        self.face_area_metric = MyMetric(name="M_FaceArea")
        self.L_strain_metric = MyMetric(name="L_Strain")
        # Bending
        self.bending_loss = BendingLoss(self.garment, self.w_bending_follow_template)
        self.bending_metric = MyMetric(name="M_Bending")
        self.L_bending_metric = MyMetric(name="L_Bending")

        # Collision
        self.collision_loss = CollisionLoss()
        self.collision_metric = MyMetric(name="M_Collision")
        self.L_collision_metric = MyMetric(name="L_Collision")

        # Gravity
        self.gravity_loss = GravityLoss(mass=self.garment.v_mass)
        self.L_gravity_metric = MyMetric(name="L_Gravity")

        # Intertia
        self.inertia_loss = InertiaLoss(self.garment.v_mass, 1 / 30)
        self.L_inertia_metric = MyMetric(name="L_Inertia")

        # Isometry
        if self.config.getboolean('collision_aware'):
            self.isometry_loss = CAIsometryLoss(self.garment, self.col_k_amp)
        else:
            self.isometry_loss = IsometryLoss(self.garment)

        self.isometry_metric = MyMetric(name='Isometry')
        self.L_isometry_metric = MyMetric(name='L_Isometry')

        # Pinning
        if self.garment.pinning:
            self.pinning_loss = PinningLoss(self.garment)
            self.L_pinning_metric = MyMetric(name='L_pinning')

    def compute_losses_and_metrics(self, vb, vg, vn, unskinned):
        vg_stacked = tf.reshape(vg, (-1, self.garment.num_vertices, 3))
        # Cloth
        if self.config['cloth_model'] == "mass-spring":
            cloth_loss, strain_metric, face_area_metric = self.cloth_loss(vg_stacked)
        elif self.config['cloth_model'] == "stvk":
            cloth_loss, strain_metric, face_area_metric = self.cloth_loss(vg_stacked)
        # Bending
        bending_loss, bending_metric = self.bending_loss(vg_stacked)
        # Collision
        collision_loss, collision_metric, distances = self.collision_loss(vg, vb, vn)
        # Gravity
        gravity_loss = self.gravity_loss(vg_stacked)
        # Inertia
        inertia_loss = self.inertia_loss(vg[:, -3:])
        # Pinning
        if self.garment.pinning:
            pinning_loss = self.pinning_loss(unskinned)

        isometry_loss = 0
        if self.config.getfloat('w_isometry') > 0:
            if self.config.getboolean('collision_aware'):
                isometry_loss, isometry_metric = self.isometry_loss(vg_stacked, distances, self.current_epoch)
            else:
                isometry_loss, isometry_metric = self.isometry_loss(vg_stacked)

        # Combine loss
        loss = (
            self.config.getfloat('w_strain') * cloth_loss
            + self.config.getfloat('w_bend') * bending_loss
            + self.config.getfloat('w_collision') * collision_loss
            + self.config.getfloat('w_gravity') * gravity_loss
            + inertia_loss
            + self.config.getfloat('w_isometry') * isometry_loss
        )

        if self.garment.pinning:
            loss += self.config.getfloat('w_pinning') * pinning_loss

        # Update metrics
        self.loss_metric.update_state(loss)
        self.strain_metric.update_state(strain_metric)
        self.face_area_metric.update_state(face_area_metric)
        self.L_strain_metric.update_state(cloth_loss)
        self.bending_metric.update_state(bending_metric)
        self.L_bending_metric.update_state(bending_loss)
        self.collision_metric.update_state(collision_metric)
        self.L_collision_metric.update_state(collision_loss)
        self.L_gravity_metric.update_state(gravity_loss)
        self.L_inertia_metric.update_state(inertia_loss)
        if self.config.getfloat('w_isometry') > 0:
            self.isometry_metric.update_state(isometry_metric)
            self.L_isometry_metric.update_state(isometry_loss)
        if self.garment.pinning:
            self.L_pinning_metric.update_state(pinning_loss)
        return loss

    def train_step(self, inputs):
        pose, shape, trans_vel, translation = inputs

        batch_size = tf.shape(pose)[0]
        hidden_states = self.prepare_hidden_state(batch_size, training=True)
        vb, _ = self.body(
            shape=tf.reshape(shape, (-1, 10)),
            pose=tf.reshape(pose, (-1, 72)),
            translation=tf.reshape(translation, (-1, 3)),
        )
        vbn = VertexNormals()(vb, self.body.faces)
        with tf.GradientTape() as tape:
            vg, hidden_states, unskinned = self([pose, shape, trans_vel, translation, hidden_states], training=True)
            loss = self.compute_losses_and_metrics(vb, vg, vbn, unskinned)

        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        return {m.name: m.result() for m in self.metrics}

    def test_step(self, inputs):
        pose, shape, trans_vel, translation = inputs
        batch_size = tf.shape(pose)[0]
        hidden_states = self.prepare_hidden_state(batch_size, training=False)
        vb, _ = self.body(
            shape=tf.reshape(shape, (-1, 10)),
            pose=tf.reshape(pose, (-1, 72)),
            translation=tf.reshape(translation, (-1, 3)),
        )
        vbn = VertexNormals()(vb, self.body.faces)
        vg, hidden_states, unskinned = self([pose, shape, trans_vel, translation, hidden_states], training=False)
        self.compute_losses_and_metrics(vb, vg, vbn, unskinned)
        return {m.name: m.result() for m in self.metrics}

    @property
    def metrics(self):
        base_metrics = [
                self.loss_metric,
                self.L_strain_metric,
                self.L_bending_metric,
                self.L_collision_metric,
                self.L_gravity_metric,
                self.L_inertia_metric,
            ]
        if self.config.getfloat('w_isometry') > 0:
            base_metrics.append(self.L_isometry_metric)
        if self.garment.pinning:
            base_metrics.append(self.L_pinning_metric)
        base_metrics += self.additional_metrics()
        return base_metrics

    def additional_metrics(self):
        base_metrics = [
            self.strain_metric,
            self.face_area_metric,
            self.bending_metric,
            self.collision_metric,
        ]
        if self.config.getfloat('w_isometry') > 0:
            base_metrics.append(self.isometry_metric)
        return base_metrics

    def predict(self, inputs, fix_collision=False, output_obj=False, output_dir='./'):
        pose, shape, trans_vel, translation = inputs
        assert pose.ndim == shape.ndim == trans_vel.ndim == translation.ndim, "Input dimensions do not match."
        if pose.ndim == 2:
            pose = tf.expand_dims(pose, axis=0)
            shape = tf.expand_dims(shape, axis=0)
            trans_vel = tf.expand_dims(trans_vel, axis=0)
            translation = tf.expand_dims(translation, axis=0)
        n_frame = pose.shape[1]
        hidden_states = self.prepare_hidden_state(batch_size=1, training=False)
        vb, _ = self.body(shape[0], pose[0], translation[0])
        pred, hidden_states, _ = self([pose, shape, trans_vel, translation, hidden_states], training=False)
        vg = pred[0].numpy()
        vb = vb.numpy()
        if fix_collision:
            vbn = VertexNormals()(vb, self.body.faces)
            for i in range(n_frame):
                vg[i] = fix_collisions(vg[i], vb[i], vbn[i].numpy())
        if output_obj:
            for i in range(n_frame):
                filename_g = os.path.join(output_dir, f"{i:04d}_{self.config['garment']}.obj")
                utils.save_obj(filename_g, vg[i], self.garment.f, rgb=(61, 133, 198))
                filename_b = os.path.join(output_dir, f"{i:04d}_body.obj")
                utils.save_obj(filename_b, vb[i], self.body.faces)
        return vg, vb

    def prepare_hidden_state(self, batch_size, training=False):
        if training:
            hidden_states = [
                tf.random.normal([batch_size, 256], mean=0, stddev=0.1, dtype=tf.float32),  # State 0
                tf.random.normal([batch_size, 256], mean=0, stddev=0.1, dtype=tf.float32),  # State 1
                tf.random.normal([batch_size, 256], mean=0, stddev=0.1, dtype=tf.float32),  # State 2
                tf.random.normal([batch_size, 256], mean=0, stddev=0.1, dtype=tf.float32),  # State 3
            ]
        else:
            hidden_states = [
                tf.zeros((batch_size, 256), dtype=tf.float32),  # State 0
                tf.zeros((batch_size, 256), dtype=tf.float32),  # State 1
                tf.zeros((batch_size, 256), dtype=tf.float32),  # State 2
                tf.zeros((batch_size, 256), dtype=tf.float32),  # State 3
            ]
        hidden_states = tf.stack(hidden_states)
        return hidden_states

    @tf.function
    def call(self, inputs, training=False):
        """
        Inputs:
            pose: body pose [batch_size, num_frame, 72]
            shape: body shape parameters [batch_size, num_frame, 10]
            trans_vel: translation velocity [batch_size, num_frame, 3]
            translation: [batch_size, num_frame, 3]
            hidden_states: initial hidden states [4, batch_size, 256]

        """
        pose, shape, trans_vel, translation, hidden_states = inputs
        num_frames = tf.shape(pose)[1]
        x = tf.concat([shape, pose, trans_vel], axis=-1)

        x, state1 = self.gru1(x, initial_state=hidden_states[0])
        x, state2 = self.gru2(x, initial_state=hidden_states[1])
        x, state3 = self.gru3(x, initial_state=hidden_states[2])
        x, state4 = self.gru4(x, initial_state=hidden_states[3])
        x = self.linear(x)

        verts_num = self.garment.num_vertices

        # [batch_size, num_frames, verts_num, 3]
        x = tf.reshape(x, [-1, verts_num, 3])

        body = self.body
        hidden_states = [state1, state2, state3, state4]

        _, joint_transforms = body(
            shape=tf.reshape(shape, (-1, 10)),
            pose=tf.reshape(pose, (-1, 72)),
            translation=tf.reshape(translation, (-1, 3))
        )
        v_garment_unskinned = x + self.garment.v_template
        v_garment_skinning_cur = smpl.LBS()(v_garment_unskinned, joint_transforms, self.skin_weight)
        v_garment_skinning_cur += tf.reshape(translation, (-1, 1, 3))
        v_garment_skinning = tf.reshape(v_garment_skinning_cur, (-1, num_frames, verts_num, 3))

        return v_garment_skinning, hidden_states, v_garment_unskinned




