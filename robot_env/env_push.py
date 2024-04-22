import numpy as np
import gym
import math
import random
import pybullet
import platform, os
from gym import utils
from robot_env.robot import robot
from gym.utils import seeding
from gym import spaces
from args import ArgumentParser

class robotGymPushEnv(gym.Env, utils.EzPickle):

    def __init__(
            self, 
            distance_threshold = 0.1, 
            isDiscrete = False,
        ):
        
        utils.EzPickle.__init__(self)
        
        self.args = ArgumentParser().parse_args()

        self.distance_threshold = distance_threshold
        self.isDiscrete = isDiscrete
        self.blockUid = -1

        if self.args.GUI:
            self.physics = pybullet.connect(pybullet.GUI)
        else:
            self.physics = pybullet.connect(pybullet.DIRECT)

        if platform.system() == "Windows":
            os.system('cls')
        elif platform.system() == "Linux":
            os.system('clear')

        self.timeStep= 1. / 240.
        self.robot = robot(task='push')
        
        action_dim = 5
        self.action_bound = 1

        if self.isDiscrete:
            self.action_space = spaces.Discrete(action_dim)
        else:
            action_high = np.array([self.action_bound] * action_dim)
            self.action_space = spaces.Box(-action_high, action_high)
        
        self._seed()
        self.init_obs = self.reset()

    def reset(self):

        pybullet.setPhysicsEngineParameter(numSolverIterations=150)
        pybullet.setTimeStep(self.timeStep)
        
        self.rest_poses = [
            0.006418, 0.413184, -0.011401, -1.589317, 0.005379, 1.137684, -0.006539,
            0.000048, -0.299912, 0.000000, -0.000043, 0.299960, 0.000000, -0.000200
        ]
        for i in range(pybullet.getNumJoints(self.robot.robotid)):
            pybullet.resetJointState(self.robot.robotid, i, self.rest_poses[i])
        
        for _ in range(100):

            xpos = 0.2 + 0.6 * random.random()
            ypos = 0.6 * random.random() * random.choice([-1, 1])
            zpos = 0.65
            ang = math.pi * 0.5 + math.pi * random.random()
            orn = pybullet.getQuaternionFromEuler([0, 0, ang])

            xpos_target = 0.2 + 0.6 * random.random()
            ypos_target = 0.6 * random.random() * random.choice([-1, 1])
            zpos_target = 0.65
            ang_target = math.pi * 0.5 + math.pi * random.random()
            orn_target = pybullet.getQuaternionFromEuler([0, 0, ang_target])
            
            self.dis_between_target_block = math.sqrt(
                (xpos-xpos_target) ** 2 +
                (ypos-ypos_target) ** 2 +
                (zpos-zpos_target) ** 2
            )

            if self.dis_between_target_block >= 0.15 and self.dis_between_target_block <= 0.6: break

        pos_object = [xpos, ypos, zpos]
        pos_target = [xpos_target, ypos_target, zpos_target]
        
        if self.blockUid == -1:
            self.blockUid = pybullet.loadURDF("urdf/cube_small_push.urdf", pos_object, orn)
            self.targetUid = pybullet.loadURDF("urdf/cube_small_target_push.urdf", pos_target, orn_target, useFixedBase=1)
        else:
            pybullet.removeBody(self.blockUid)
            pybullet.removeBody(self.targetUid)
            self.blockUid = pybullet.loadURDF("urdf/cube_small_push.urdf", pos_object, orn)
            self.targetUid = pybullet.loadURDF("urdf/cube_small_target_push.urdf", pos_target, orn_target, useFixedBase=1)
        
        pybullet.setCollisionFilterPair(self.targetUid, self.blockUid, -1, -1, 0)
        pybullet.setGravity(0, 0, -9.8)

        self.goal=np.array(pos_target)
        self._observation = self._getObservation()
        
        return self._observation

    def step(self, action):
        
        self._apply_action(action)

        pybullet.stepSimulation()
        
        obs = self._getObservation()
        reward = self._reward(obs)
        done = self._is_success(obs['block_pos'], self.goal)
        info = {'is_success': self._is_success(obs['block_pos'], self.goal)}
        reward = self._reward(obs)

        return obs, reward, done, info

    def _apply_action(self, action):
        self.robot.execute_action(action)
    
    def _getObservation(self):       

        self._observation = []
        state = pybullet.getLinkState(self.robot.robotid, self.robot.gripperIdx)
        pos, orn = state[0], state[1]
        euler = pybullet.getEulerFromQuaternion(orn)
        self._observation.extend(list(pos))
        self._observation.extend(list(euler))

        gripperState = pybullet.getLinkState(self.robot.robotid, self.robot.gripperIdx)
        gripperPos = gripperState[0]
        gripperOrn = gripperState[1]

        blockPos, blockOrn = pybullet.getBasePositionAndOrientation(self.blockUid)
        targetPos, targetOrn = pybullet.getBasePositionAndOrientation(self.targetUid)
        invGripperPos, invGripperOrn = pybullet.invertTransform(gripperPos, gripperOrn)
        
        # block rel to gripper: list: rel_x, rel_y, eul_z
        blockPosInGripper, blockOrnInGripper = pybullet.multiplyTransforms(invGripperPos, invGripperOrn, blockPos, blockOrn)
        blockEulerInGripper = pybullet.getEulerFromQuaternion(blockOrnInGripper)
        blockInGripperPosXYEulZ = [blockPosInGripper[0], blockPosInGripper[1], blockEulerInGripper[2]]
        
        # target rel to gripper: list: rel_x, rel_y, eul_z
        targetPosInGripper, targetOrnInGripper = pybullet.multiplyTransforms(invGripperPos, invGripperOrn, targetPos, targetOrn)
        targetEulerInGripper = pybullet.getEulerFromQuaternion(targetOrnInGripper)
        targetInGripperPosXYEulZ = [targetPosInGripper[0], targetPosInGripper[1], targetEulerInGripper[2]]
        
        # target rel to block: list: rel_x, rel_y, eul_z      
        targetPosInBlock, targetOrnInBlock = pybullet.multiplyTransforms(blockPos, blockOrn, targetPos, targetOrn)
        targetEulerInBlock = pybullet.getEulerFromQuaternion(targetOrnInBlock)
        targetInBlockPosXYEulZ = [targetPosInBlock[0], targetPosInBlock[1], targetEulerInBlock[2]]

        self._observation.extend(list(blockInGripperPosXYEulZ))
        self._observation.extend(list(targetInGripperPosXYEulZ))
        self._observation.extend(list(targetInBlockPosXYEulZ))
        self._observation = np.array(self._observation).reshape(-1)

        self._achieved_goal = self._observation[:3]
        self._desired_goal = np.array(targetPos).flatten()
        self._block_pos = np.array(blockPos).flatten()

        contacts = pybullet.getContactPoints(bodyA=self.robot.robotid, bodyB=self.robot.tableUid)
        if contacts:
            collision_withTable = np.array(1)
        else:
            collision_withTable = np.array(0)
        contacts = pybullet.getContactPoints(bodyA=self.robot.robotid, bodyB=self.blockUid)
        if contacts:
            collision_withBlock = np.array(1)
        else:
            collision_withBlock = np.array(0)
        
        stateL = pybullet.getLinkState(self.robot.robotid, self.robot.clawLIdx)
        stateR = pybullet.getLinkState(self.robot.robotid, self.robot.clawRIdx)
        posL, posR = np.array(stateL[0]), np.array(stateR[0])
        gripperCenterPos = np.array(np.mean([posL, posR], axis=0))
        
        return {
            'observation': self._observation.copy(),
            'achieved_goal': self._achieved_goal.copy(),
            'desired_goal': self._desired_goal.copy(),
            'block_pos': self._block_pos.copy(),
            'gripperCenterPos': gripperCenterPos.copy(),
            'collision_withTable': collision_withTable.copy(),
            'collision_withBlock': collision_withBlock.copy()
        }
    
    def _reward(self, observation):

        desired_goal = observation['desired_goal']
        block_pos = observation['block_pos']
        gripperCenterPos = observation['gripperCenterPos']
        collision_withTable = observation['collision_withTable']

        L2_dist_centerToBlock = self._euler_dist(gripperCenterPos, block_pos)
        L2_dist_blockToTarget = self._euler_dist(block_pos, desired_goal)

        # time & dist punishment
        reward = -1 * L2_dist_centerToBlock

        # collision punishment
        if collision_withTable:
            reward = reward - 0.01
        
        # grasp reward
        if L2_dist_centerToBlock < self.distance_threshold:
            reward = reward + 100
        
        # success reward
        if L2_dist_blockToTarget < self.distance_threshold:
            reward = reward + 1000

        return np.array([reward])

    def _is_success(self, block_pos, desired_goal):

        d = self._euler_dist(block_pos, desired_goal)
        return (d < self.distance_threshold).astype(np.float32)
    
    def _seed(self, seed=None):

        self.np_random, seed = seeding.np_random(seed)
        return [seed]
    
    def _euler_dist(self, goal_a, goal_b):

        assert goal_a.shape == goal_b.shape

        return np.linalg.norm(goal_a - goal_b, axis=-1)