import numpy as np
import torch

from utils import Isometry3d, to_rotation, to_tensor, to_numpy, DEVICE, DTYPE



class Feature(object):
    # id for next feature
    next_id = 0

    # Takes a vector from the cam0 frame to the cam1 frame.
    R_cam0_cam1 = None
    t_cam0_cam1 = None

    def __init__(self, new_id=0, optimization_config=None):
        # An unique identifier for the feature.
        self.id = new_id

        # Store the observations of the features in the
        # state_id(key)-image_coordinates(value) manner.
        self.observations = dict()   # <StateID, vector4d>

        # 3d postion of the feature in the world frame.
        self.position = torch.zeros(3, device=DEVICE, dtype=DTYPE)

        # A indicator to show if the 3d postion of the feature
        # has been initialized or not.
        self.is_initialized = False

        # Optimization configuration for solving the 3d position.
        self.optimization_config = optimization_config

    def cost(self, T_c0_ci, x, z):
        """
        Compute the cost of the camera observations

        Arguments:
            T_c0_c1: A rigid body transformation takes a vector in c0 frame 
                to ci frame. (Isometry3d)
            x: The current estimation. (vec3 tensor)
            z: The ith measurement of the feature j in ci frame. (vec2 tensor)

        Returns:
            e: The cost of this observation. (tensor scalar)
        """
        x = to_tensor(x)
        z = to_tensor(z)
        
        # Compute hi1, hi2, and hi3 as Equation (37).
        alpha, beta, rho = x[0], x[1], x[2]
        h = T_c0_ci.R @ torch.stack([alpha, beta, torch.ones_like(alpha)]) + rho * T_c0_ci.t

        # Predict the feature observation in ci frame.
        z_hat = h[:2] / h[2]

        # Compute the residual.
        e = ((z_hat - z)**2).sum()
        return e

    def jacobian(self, T_c0_ci, x, z):
        """
        Compute the Jacobian of the camera observation

        Arguments:
            T_c0_c1: A rigid body transformation takes a vector in c0 frame 
                to ci frame. (Isometry3d)
            x: The current estimation. (vec3 tensor)
            z: The ith measurement of the feature j in ci frame. (vec2 tensor)

        Returns:
            J: The computed Jacobian. (Matrix23 tensor)
            r: The computed residual. (vec2 tensor)
            w: Weight induced by huber kernel. (float)
        """
        x = to_tensor(x)
        z = to_tensor(z)
        
        # Compute hi1, hi2, and hi3 as Equation (37).
        alpha, beta, rho = x[0], x[1], x[2]
        h = T_c0_ci.R @ torch.stack([alpha, beta, torch.ones_like(alpha)]) + rho * T_c0_ci.t
        h1, h2, h3 = h[0], h[1], h[2]

        # Compute the Jacobian.
        W = torch.zeros(3, 3, device=x.device, dtype=x.dtype)
        W[:, :2] = T_c0_ci.R[:, :2]
        W[:, 2] = T_c0_ci.t

        J = torch.zeros(2, 3, device=x.device, dtype=x.dtype)
        J[0] = W[0]/h3 - W[2]*h1/(h3*h3)
        J[1] = W[1]/h3 - W[2]*h2/(h3*h3)

        # Compute the residual.
        z_hat = torch.stack([h1/h3, h2/h3])
        r = z_hat - z

        # Compute the weight based on the residual.
        e = torch.norm(r)
        if e <= self.optimization_config.huber_epsilon:
            w = 1.0
        else:
            w = self.optimization_config.huber_epsilon / (2*e.item())

        return J, r, w

    def generate_initial_guess(self, T_c1_c2, z1, z2):
        """
        Compute the initial guess of the feature's 3d position using 
        only two views.

        Arguments:
            T_c1_c2: A rigid body transformation taking a vector from c2 frame 
                to c1 frame. (Isometry3d)
            z1: feature observation in c1 frame. (vec2 tensor)
            z2: feature observation in c2 frame. (vec2 tensor)

        Returns:
            p: Computed feature position in c1 frame. (vec3 tensor)
        """
        z1 = to_tensor(z1)
        z2 = to_tensor(z2)
        
        # Construct a least square problem to solve the depth.
        one = torch.ones(1, device=z1.device, dtype=z1.dtype)
        m = T_c1_c2.R @ torch.cat([z1, one])
        a = m[:2] - z2*m[2]                   # vec2
        b = z2*T_c1_c2.t[2] - T_c1_c2.t[:2]   # vec2

        # Solve for the depth.
        depth = torch.dot(a, b) / torch.dot(a, a)
        
        p = torch.cat([z1, one]) * depth
        return p

    def check_motion(self, cam_states):
        """
        Check the input camera poses to ensure there is enough translation 
        to triangulate the feature

        Arguments:
            cam_states: input camera poses. (dict of <CAMStateID, CAMState>)

        Returns:
            True if the translation between the input camera poses 
                is sufficient. (bool)
        """
        if self.optimization_config.translation_threshold < 0:
            return True

        observation_ids = list(self.observations.keys())
        first_id = observation_ids[0]
        last_id = observation_ids[-1]

        first_cam_pose = Isometry3d(
            to_rotation(cam_states[first_id].orientation).T,
            cam_states[first_id].position)

        last_cam_pose = Isometry3d(
            to_rotation(cam_states[last_id].orientation).T,
            cam_states[last_id].position)

        # Get the direction of the feature when it is first observed.
        # This direction is represented in the world frame.
        obs = to_tensor(self.observations[first_id][:2])
        one = torch.ones(1, device=obs.device, dtype=obs.dtype)
        feature_direction = torch.cat([obs, one])
        feature_direction = feature_direction / torch.norm(feature_direction)
        feature_direction = first_cam_pose.R @ feature_direction

        # Compute the translation between the first frame and the last frame. 
        # We assume the first frame and the last frame will provide the 
        # largest motion to speed up the checking process.
        translation = last_cam_pose.t - first_cam_pose.t
        parallel = torch.dot(translation, feature_direction)
        orthogonal_translation = translation - parallel * feature_direction

        return (torch.norm(orthogonal_translation).item() > 
            self.optimization_config.translation_threshold)

    def initialize_position(self, cam_states):
        """
        Intialize the feature position based on all current available 
        measurements.

        The computed 3d position is used to set the position member variable. 
        Note the resulted position is in world frame.

        Arguments:
            cam_states: A dict containing the camera poses with its ID as the 
                associated key value. (dict of <CAMStateID, CAMState>)

        Returns:
            True if the estimated 3d position of the feature is valid. (bool)
        """
        cam_poses = []     # [Isometry3d]
        measurements = []  # [vec2 tensor]

        T_cam1_cam0 = Isometry3d(
            Feature.R_cam0_cam1, Feature.t_cam0_cam1).inverse()

        for cam_id, m in self.observations.items():
            try:
                cam_state = cam_states[cam_id]
            except KeyError:
                continue
            
            # Add measurements.
            m_tensor = to_tensor(m)
            measurements.append(m_tensor[:2])
            measurements.append(m_tensor[2:])

            # This camera pose will take a vector from this camera frame
            # to the world frame.
            cam0_pose = Isometry3d(
                to_rotation(cam_state.orientation).T, cam_state.position)
            cam1_pose = cam0_pose * T_cam1_cam0

            cam_poses.append(cam0_pose)
            cam_poses.append(cam1_pose)

        # All camera poses should be modified such that it takes a vector 
        # from the first camera frame in the buffer to this camera frame.
        T_c0_w = cam_poses[0]
        cam_poses_tmp = []
        for pose in cam_poses:
            cam_poses_tmp.append(pose.inverse() * T_c0_w)
        cam_poses = cam_poses_tmp

        # Generate initial guess
        initial_position = self.generate_initial_guess(
            cam_poses[-2], measurements[0], measurements[-2])
        one = torch.ones(1, device=initial_position.device, dtype=initial_position.dtype)
        solution = torch.cat([initial_position[:2], one]) / initial_position[2]

        # Apply Levenberg-Marquart method to solve for the 3d position.
        lambd = self.optimization_config.initial_damping
        inner_loop_count = 0
        outer_loop_count = 0
        is_cost_reduced = False
        delta_norm = float('inf')

        # Compute the initial cost.
        total_cost = torch.tensor(0.0, device=DEVICE, dtype=DTYPE)
        for cam_pose, measurement in zip(cam_poses, measurements):
            total_cost = total_cost + self.cost(cam_pose, solution, measurement)

        # Outer loop.
        while (outer_loop_count < 
            self.optimization_config.outer_loop_max_iteration
            and delta_norm > 
            self.optimization_config.estimation_precision):

            A = torch.zeros(3, 3, device=DEVICE, dtype=DTYPE)
            b = torch.zeros(3, device=DEVICE, dtype=DTYPE)
            for cam_pose, measurement in zip(cam_poses, measurements):
                J, r, w = self.jacobian(cam_pose, solution, measurement)
                if w == 1.0:
                    A = A + J.T @ J
                    b = b + J.T @ r
                else:
                    A = A + w * w * J.T @ J
                    b = b + w * w * J.T @ r

            # Inner loop.
            # Solve for the delta that can reduce the total cost.
            while (inner_loop_count < 
                self.optimization_config.inner_loop_max_iteration
                and not is_cost_reduced):

                identity = torch.eye(3, device=DEVICE, dtype=DTYPE)
                delta = torch.linalg.solve(A + lambd * identity, b)   # vec3
                new_solution = solution - delta
                delta_norm = torch.norm(delta).item()

                new_cost = torch.tensor(0.0, device=DEVICE, dtype=DTYPE)
                for cam_pose, measurement in zip(cam_poses, measurements):
                    new_cost = new_cost + self.cost(
                        cam_pose, new_solution, measurement)

                if new_cost < total_cost:
                    is_cost_reduced = True
                    solution = new_solution
                    total_cost = new_cost
                    lambd = max(lambd/10., 1e-10)
                else:
                    is_cost_reduced = False
                    lambd = min(lambd*10., 1e12)
                
                inner_loop_count += 1
            inner_loop_count = 0
            outer_loop_count += 1

        # Covert the feature position from inverse depth
        # representation to its 3d coordinate.
        one = torch.ones(1, device=solution.device, dtype=solution.dtype)
        final_position = torch.cat([solution[:2], one]) / solution[2]

        # Check if the solution is valid. Make sure the feature
        # is in front of every camera frame observing it.
        is_valid_solution = True
        for pose in cam_poses:
            position = pose.R @ final_position + pose.t
            if position[2] <= 0:
                is_valid_solution = False
                break

        # Convert the feature position to the world frame.
        self.position = T_c0_w.R @ final_position + T_c0_w.t

        self.is_initialized = is_valid_solution
        return is_valid_solution