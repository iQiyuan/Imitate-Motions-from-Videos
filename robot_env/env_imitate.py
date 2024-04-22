import numpy as np
import gym
import math
import pybullet
import os, inspect
import platform
import textwrap
from gym import utils
from scipy.interpolate import CubicSpline
from statsmodels.nonparametric.smoothers_lowess import lowess
from robot_env.robot import robot
from gym.utils import seeding
from gym import spaces
from args import ArgumentParser

class robotGymImitateEnv(gym.Env, utils.EzPickle):

    def __init__(self):
        
        utils.EzPickle.__init__(self)
        
        # parse arguments
        self.args = ArgumentParser().parse_args()
        self.is_discrete = self.args.is_discrete
        self.distance_threshold = self.args.distance_threshold
        self.video = self.args.video
        self.enable_smoothing = self.args.enable_smoothing
        self.enable_reference_model = self.args.enable_reference_model

        # inverse kinematics & null space control
        self.ref_motion_use_ik = True
        self.ll = [-2.96, -2.09, -2.96, -2.09, -2.96, -2.09, -3.05]
        self.ul = [+2.96, +2.09, +2.96, +2.09, +2.96, +2.09, +3.05]
        self.jr = [5.8, 4.0, 5.8, 4.0, 5.8, 4.0, 6.0]
        self.rp = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self.dp = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]

        # capture env changes
        self.step_counter = 0
        self.ref_idx = 0
        self.successful_imitate_count = 0
        self.endEff_idx = 6

        # velocity tendence parameter
        self.prev_ref_anglPos = None
        self.prev_rbt_anglPos = None
        
        # smoothing parameters
        self.interpolated_quantity = 150 #recommend >= 20
        self.exp_alpha = 0.03
        self.smoothed_action = None

        USE_GUI = self.args.GUI
        
        if USE_GUI:
            self.physics = pybullet.connect(pybullet.GUI)
        else:
            self.physics = pybullet.connect(pybullet.DIRECT)

        if platform.system() == "Windows":
            os.system('cls')
        elif platform.system() == "Linux":
            os.system('clear')

        self.timeStep = 1. / 240.
        pybullet.setTimeStep(self.timeStep)

        self.robot = robot(task='imitate')

        self.ref_robot = self._generate_refModel()
        self.ref_motion = self._generate_refMotion()
        
        # ACTION: joint position control
        self.action_dim = 6
        self.action_bound = 2 * math.pi

        if self.is_discrete:
            self.action_space = spaces.Discrete(self.action_dim)
        else:
            act = np.array([self.action_bound] * self.action_dim)
            self.action_space = spaces.Box(-act, act)

        self.seed = self._seed()
        self.init_obs = self.reset()

    def reset(self):

        self.step_counter = 0
        self.successful_imitate_count = 0

        self._update_reference(self.ref_robot)

        pybullet.setPhysicsEngineParameter(numSolverIterations=150)
        pybullet.setTimeStep(self.timeStep)
        pybullet.setGravity(0, 0, -9.8)

        return self._get_observation()
    
    def step(self, action):

        self._execute_exp_soothed_act(action)

        for i in range(20):
            pybullet.stepSimulation()

        obs = self._get_observation()
        done = self._is_terminated(obs)
        rwd = self._reward(obs)
        info = {'is_success': self._is_success(obs)}       

        self.step_counter = self.step_counter + 1 
        
        self._update_reference(self.ref_robot)

        return obs, rwd, done, info

    def _get_observation(self):
        
        ref_anglPos, rbt_anglPos = np.array([]), np.array([])

        # joint positions of 6 motors
        for idx in range(self.endEff_idx):

            ref_anglPos = np.append(ref_anglPos, pybullet.getJointState(self.ref_robot, idx)[0])
            rbt_anglPos = np.append(rbt_anglPos, pybullet.getJointState(self.robot.robotid, idx)[0])

        # end-effector position
        ref_edefPos = np.array(pybullet.getLinkState(self.ref_robot, self.endEff_idx)[0])
        rbt_edefPos = np.array(pybullet.getLinkState(self.robot.robotid, self.endEff_idx)[0])

        # angular velocity as tendency of future motion for better imitation
        if self.prev_ref_anglPos is None and self.prev_rbt_anglPos is None:
            ref_anglVel = ref_anglPos
            rbt_anglVel = rbt_anglPos
        else:
            ref_anglVel = ref_anglPos - self.prev_ref_anglPos
            rbt_anglVel = rbt_anglPos - self.prev_rbt_anglPos

        self.prev_ref_anglPos = ref_anglPos
        self.prev_rbt_anglPos = rbt_anglPos
        
        angldiff = ref_anglPos - rbt_anglPos
        velcdiff = ref_anglVel - rbt_anglVel
        edefdiff = ref_edefPos - rbt_edefPos
        
        env_obs = np.array([])
        env_obs = np.append(env_obs, ref_anglPos)
        env_obs = np.append(env_obs, angldiff)
        env_obs = np.append(env_obs, velcdiff)
        env_obs = np.append(env_obs, edefdiff)
        env_obs = self._normalize_obs(env_obs)

        obs = {
            'observation': env_obs,
            'ref_anglPos': ref_anglPos, 'rbt_anglPos': rbt_anglPos,
            'ref_anglVel': ref_anglVel, 'rbt_anglVel': rbt_anglVel,
            'ref_edefPos': ref_edefPos, 'rbt_edefPos': rbt_edefPos
        }

        return obs

    def _execute_exp_soothed_act(self, action):
        
        # smoothing action de-noising ppo Gaussian sampled action
        if self.smoothed_action is None:
            self.smoothed_action = action
        else:
            self.smoothed_action = self.exp_alpha * action + (1 - self.exp_alpha) * self.smoothed_action
        
        self.robot.execute_action(self.smoothed_action)

    def _reward(self, obs):

        ref_anglPos, rbt_anglPos = obs['ref_anglPos'], obs['rbt_anglPos']
        ref_anglVel, rbt_anglVel = obs['ref_anglVel'], obs['rbt_anglVel']
        ref_edefPos, rbt_edefPos = obs['ref_edefPos'], obs['rbt_edefPos']

        coefficients = [0.1, 10, 1, 1, 10, 10]
        
        diff_anglPos = np.abs(ref_anglPos - rbt_anglPos)
        diff_anglVel = np.abs(ref_anglVel - rbt_anglVel)

        weighted_diff_anglPos = [diff_anglPos[i] * coefficients[i] for i in range(len(diff_anglPos))]
        weighted_diff_anglVel = [diff_anglVel[i] * coefficients[i] for i in range(len(diff_anglVel))]

        rwd_anglPos = np.sum(weighted_diff_anglPos)
        rwd_anglVel = np.sum(weighted_diff_anglVel)
        rwd_edefPos = np.linalg.norm(ref_edefPos - rbt_edefPos)

        anglPos_weight = 1.0
        anglVel_weight = 0.1
        edefPos_weight = 100.0

        reward = (
            -anglPos_weight * rwd_anglPos 
            -anglVel_weight * rwd_anglVel
            -edefPos_weight * rwd_edefPos
        )

        return np.array([reward])

    def _is_terminated(self, obs):
        
        ref = obs['ref_anglPos']
        rbt = obs['rbt_anglPos']

        sum_diff_anglPos = np.sum(np.abs([ref - rbt]))

        if sum_diff_anglPos < self.distance_threshold:
            self.successful_imitate_count += 1
            
            if self.successful_imitate_count / self.ref_motion.shape[0] >= 0.9:
                self.successful_imitate_count = 0
                print("Successful imitate! Rate: > 90%")
                return True
        
        return False

    def _is_success(self, obs):

        ref = obs['ref_anglPos']
        rbt = obs['rbt_anglPos']

        if np.sum(np.abs([ref - rbt])) < self.distance_threshold:
            return True
        else:
            return False
        
    def _update_reference(self, ref_robot):
        
        ref_motion = self.ref_motion
        ref_idx = self.ref_idx

        ref_pose = ref_motion[ref_idx, :]
        self.ref_idx = self.ref_idx + 1

        if self.ref_idx >= self.ref_motion.shape[0]:
            self.ref_idx = 0

        if self.enable_reference_model:
            for i in range(ref_pose.shape[0]):
                pybullet.resetJointState(ref_robot, i, ref_pose[i])
    
    def _generate_refModel(self):

        ref_robot = pybullet.loadURDF("kuka_iiwa/model.urdf")
        
        pybullet.resetBasePositionAndOrientation(ref_robot, [-0.25, 0.0, 0.6], [0, 0, 0, 1])
        pybullet.changeDynamics(ref_robot, -1, linearDamping=0, angularDamping=0)
        pybullet.setCollisionFilterGroupMask(ref_robot, -1, collisionFilterGroup=0, collisionFilterMask=0)
        pybullet.changeVisualShape(ref_robot, -1, rgbaColor=[1, 1, 1, 0.8])
        pybullet.changeDynamics(
            ref_robot, -1,
            activationState = pybullet.ACTIVATION_STATE_SLEEP +
            pybullet.ACTIVATION_STATE_ENABLE_SLEEPING +
            pybullet.ACTIVATION_STATE_DISABLE_WAKEUP
        )

        # disable sleeping
        for idx in range(pybullet.getNumJoints(ref_robot)):
            pybullet.setCollisionFilterGroupMask(ref_robot,idx,collisionFilterGroup=0,collisionFilterMask=0)
            pybullet.changeVisualShape(ref_robot, idx, rgbaColor=[1, 1, 1, 0.5])
            pybullet.changeDynamics(
                ref_robot, idx,
                activationState=pybullet.ACTIVATION_STATE_SLEEP +
                pybullet.ACTIVATION_STATE_ENABLE_SLEEPING +
                pybullet.ACTIVATION_STATE_DISABLE_WAKEUP
            )

        return ref_robot
    
    def _generate_refMotion(self):

        info_str = """
        ———————————————————————————————————————————————————————————————
        Generating & Smoothing reference motion...
        """
        info_str = textwrap.dedent(info_str)
        print(info_str)

        dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
        dir = os.path.os.path.dirname(dir)
        kpts_dir = dir + '/demo_data/' + self.video + '/keypoints3D.npz'
        kpts = np.load(kpts_dir, allow_pickle=True)['reconstruction'][:, 3, :].reshape(-1, 3)
        
        # Cubic interpolated enlarge dataset
        if kpts.shape[0] < 2048:
            index = np.arange(kpts.shape[0])
            interp_func = CubicSpline(index, kpts, axis=0)
            new_index = np.linspace(0, index[-1], 2048)
            kpts = interp_func(new_index)

        ref_poses = np.array([])
        
        if self.ref_motion_use_ik:
            print("Reference Motion Generated using NullSpace IK.")
        else:
            print("Reference Motion Generated using Regulaar IK.")
        
        for frame in range(kpts.shape[0]):
            pos = kpts[frame] + np.array([-0.25, 0.0, 0.6])
            joint_pos = self._solve_ik(self.ref_robot,pos)[:6]
            ref_poses = np.append(ref_poses, joint_pos)
        
        frame = int(ref_poses.shape[0] / 6)
        ref_poses = ref_poses.reshape(frame, 6)

        # Cubic Spline interpolate enlarges sparse components of demo dataset
        sparse_ref_index = np.where(np.linalg.norm(ref_poses[1:] - ref_poses[:-1], axis=1)>0.5)[0]

        shape_0 = ref_poses.shape[0] + self.interpolated_quantity * len(sparse_ref_index)
        interpolated_ref = np.zeros((shape_0, 6))

        for col in range(ref_poses.shape[1]):
            data = ref_poses[:, col]
            idx_range = np.arange(data.shape[0])
            cubic_interp = CubicSpline(idx_range, data)

            for idx in reversed(sparse_ref_index):
                inserted_index = np.linspace(idx, idx+1, self.interpolated_quantity)
                inserted_value = cubic_interp(inserted_index)
                data = np.insert(data, idx+1, inserted_value)
            
            interpolated_ref[:, col] = data

        # lowess smoothing raw data
        smoothed_ref = np.zeros_like(interpolated_ref)
        for i in range(interpolated_ref.shape[1]):
            smoothed_ref[:, i] = lowess(interpolated_ref[:, i], np.arange(len(interpolated_ref[:, i])), frac=0.05, return_sorted=False)
        
        info_str = """
        Reference motion generated. Shape:      {0}
        ———————————————————————————————————————————————————————————————
        """.format(smoothed_ref.shape)
        info_str = textwrap.dedent(info_str)
        print(info_str)

        return smoothed_ref
    
    def _solve_ik(self, ref_robot, pos):
        
        if self.ref_motion_use_ik:
            joint_poses = pybullet.calculateInverseKinematics(
                ref_robot, self.endEff_idx, pos,
                lowerLimits=self.ll, upperLimits=self.ul, jointRanges=self.jr, restPoses=self.rp, jointDamping=self.dp
            )
            self.rp = joint_poses[:7]
        else:
            joint_poses = pybullet.calculateInverseKinematics(ref_robot, self.endEff_idx, pos)
        
        return joint_poses

    def render(self):

        width, height = 1080, 1080
        aspect = width / height
        fov, near, far = 60, 0.5, 15

        projection_matrix = pybullet.computeProjectionMatrixFOV(fov, aspect, near, far)
        view_matrix = pybullet.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=[-0.25, 0.0, 0.6],
            distance=2.5,
            yaw=45, pitch=-40, roll=0,
            upAxisIndex=2
        )

        width, height, rgbImg, depthImg, segImg = pybullet.getCameraImage(
            width, height, 
            view_matrix, projection_matrix, shadow=True, 
            lightDirection=[1, 1, 1],
            renderer=pybullet.ER_BULLET_HARDWARE_OPENGL
        )

        rgbImg = np.reshape(rgbImg, (height, width, 4))[:, :, :3]

        return rgbImg
    
    def _radian_dist(self, quaternion_array1, quaternion_array2):

        radian_dist = np.array([])

        for orn1, orn2 in zip(quaternion_array1, quaternion_array2):
            dist = self._solve_angle(orn1, orn2)
            radian_dist = np.append(radian_dist, dist)
        
        return radian_dist
    
    def _solve_angle(self, quaternion_orn1, quaternion_orn2):
        
        dot_product = np.dot(quaternion_orn1, quaternion_orn2)
        dot_product = np.clip(dot_product, -1.0, 1.0)
        angle = 2 * np.arccos(np.abs(dot_product))
        
        return angle

    def _normalize_obs(self, obs):

        mean = np.mean(obs)
        std = np.std(obs)
        normalized_obs = (obs - mean) / std

        return normalized_obs
    
    def _seed(self, seed = None):
        self.np_random, self.random_seed = seeding.np_random(seed)
        return self.random_seed
    
    def close(self):
        pybullet.disconnect()