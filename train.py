#!/usr/bin/env python
from ppo.ppo_imitate import train
from args import ArgumentParser


""" references:

    https://github.com/openai/baselines/tree/master/baselines
    https://github.com/nikhilbarhate99/PPO-PyTorch
    https://github.com/Vegetebird/StridedTransformer-Pose3D
    https://github.com/erwincoumans/motion_imitation
    https://github.com/bulletphysics/bullet3/tree/master"""


if __name__ == '__main__':

    args = ArgumentParser().parse_args()
    task = args.task

    train(task)
