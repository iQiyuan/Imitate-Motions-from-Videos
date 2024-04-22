import pybullet
import math
import pybullet_data
import os, inspect
import numpy as np
from args import ArgumentParser
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0, parentdir)

class robot(object):
    
    def __init__(
            self, 
            urdfRootPath=pybullet_data.getDataPath(), 
            task='imitate',
        ):

        self.args = ArgumentParser().parse_args()
        self.use_simulation = self.args.use_simulation
        self.use_orientation = self.args.use_orientation
        
        self.urdfRootPath = urdfRootPath

        self.task = task
        if self.task == 'imitate':
            self.use_inverse_kinematics = False
        elif self.task == 'push':
            self.use_inverse_kinematics = True

        self.endEffector= 6
        self.gripperIdx = 7
        self.clawLIdx = 10
        self.clawRIdx = 13

        self.rbt_base_pos = [-0.25, 0.0, 0.6]
        self.rbt_base_orn = [0, 0, 0, 1]
        
        self.reset()

    def reset(self):

        pybullet.setAdditionalSearchPath(pybullet_data.getDataPath())
        pybullet.resetSimulation()

        if self.task == 'imitate':
            self.robotid = pybullet.loadURDF("kuka_iiwa/model.urdf")
        elif self.task == 'push':
            self.robotid = pybullet.loadSDF("kuka_iiwa/kuka_with_gripper2.sdf")[0]

        pybullet.loadURDF("plane.urdf", [0, 0, 0])
        pybullet.resetBasePositionAndOrientation(self.robotid, self.rbt_base_pos, self.rbt_base_orn)

        if self.task == 'push':
            
            self.tableUid = pybullet.loadURDF(
                os.path.join(self.urdfRootPath, "table/table.urdf"), 
                basePosition=[0.5, 0.0, 0.0], 
                baseOrientation=[0, 0, 1, 1], 
                globalScaling=1
            )

            self.restPose = [
                0.006418, 0.413184, -0.011401, -1.589317, 0.005379, 1.137684, -0.006539,
                0.000048, -0.299912, 0.000000, -0.000043, 0.299960, 0.000000, -0.000200
            ]

            for idx in range(pybullet.getNumJoints(self.robotid)):
                pybullet.resetJointState(self.robotid, idx, self.restPose[idx])

        self.endEffectorPos = [0, 0, 0]
        self.endEffectorAngle = 0
    
    def execute_action(self, action):
        
        if self.use_inverse_kinematics:
            
            dx = action[0]
            dy = action[1]
            dz = action[2]

            d_endEffAng = action[3]
            d_clawOpenAng = action[4]

            realEndEffPos = pybullet.getLinkState(self.robotid, self.endEffector)[0]
                        
            self.endEffectorPos[0] = np.clip(realEndEffPos[0] + dx, 0.2, 0.8)
            self.endEffectorPos[1] = np.clip(realEndEffPos[1] + dy, -0.9, 0.9)
            self.endEffectorPos[2] = np.clip(realEndEffPos[2] + dz, 0.9, 1.0)
            
            self.endEffectorAngle = self.endEffectorAngle + d_endEffAng

            pos = self.endEffectorPos
            orn = pybullet.getQuaternionFromEuler([0, -math.pi, 0])

            jointPoses = self._solve_ik(pos, orn)
            self._execute_motion(jointPoses, self.endEffectorAngle, d_clawOpenAng)
        
        else:
            for act_idx in range(len(action)):
                
                if self.use_simulation:
                    pybullet.setJointMotorControl2(
                        self.robotid,
                        act_idx,
                        pybullet.POSITION_CONTROL,
                        targetPosition=action[act_idx],
                    )

                else:
                    pybullet.resetJointState(self.robotid, act_idx, action[act_idx])

    def _execute_motion(self, jointPoses, endEffAng, fingerAng):
        
        if self.use_simulation:
            for i in range(self.endEffector + 1):
                pybullet.setJointMotorControl2(
                    bodyUniqueId=self.robotid,
                    jointIndex=i,
                    controlMode=pybullet.POSITION_CONTROL,
                    targetPosition=jointPoses[i],
                )
        
        else:
            for i in range(self.numJoints):
                pybullet.resetJointState(self.robotid, i, jointPoses[i])

        pybullet.setJointMotorControl2(
            self.robotid,
            jointIndex=7,
            controlMode=pybullet.POSITION_CONTROL,
            targetPosition=endEffAng,
        )

        pybullet.setJointMotorControl2(
            self.robotid,
            jointIndex=8,
            controlMode=pybullet.POSITION_CONTROL,
            targetPosition=-fingerAng,
        )

        pybullet.setJointMotorControl2(
            self.robotid,
            jointIndex=11,
            controlMode=pybullet.POSITION_CONTROL,
            targetPosition=fingerAng,
        )

        pybullet.setJointMotorControl2(
            self.robotid,
            jointIndex=10,
            controlMode=pybullet.POSITION_CONTROL,
            targetPosition=0,
        )
        
        pybullet.setJointMotorControl2(
            self.robotid,
            jointIndex=13,
            controlMode=pybullet.POSITION_CONTROL,
            targetPosition=0,
        )
    
    def _solve_ik(self, pos, orn):

        if self.use_orientation:
            jointPoses = pybullet.calculateInverseKinematics(self.robotid, self.endEffector, pos, orn)

        else:
            jointPoses = pybullet.calculateInverseKinematics(self.robotid, self.endEffector, pos)
        
        return jointPoses