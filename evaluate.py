import time
import pybullet
import numpy as np
import pandas as pd
from robot_env.env_imitate import robotGymImitateEnv
from robot_env.env_push import robotGymPushEnv
from ppo.ppo import PPO
from args import ArgumentParser

def similarity_evaluate(env, evaluate_buffer_angl):
    
    similarity = 0
    for idx in range(env.endEff_idx):
        rbt = pybullet.getJointState(env.robot.robotid, idx)[0]
        ref = pybullet.getJointState(env.ref_robot, idx)[0]
        error = rbt - ref
        evaluate_buffer_angl['angl_diff_'+str(idx)].append(error)
    similarity += error
    
    return similarity

def endeff_evaluate(env, evaluate_buffer_edef):
    
    rbt = np.array(pybullet.getLinkState(env.robot.robotid, env.endEff_idx)[0])
    ref = np.array(pybullet.getLinkState(env.ref_robot, env.endEff_idx)[0])
    dist = np.linalg.norm(rbt - ref)
    evaluate_buffer_edef['edef_dist'].append(dist)

    return dist

def mpjpe_evaluate(env, evaluate_buffer_pose):
    
    dist_mpjpe = 0
    for idx in range(env.endEff_idx):
        rbt = np.array(pybullet.getLinkState(env.robot.robotid, idx)[0])
        ref = np.array(pybullet.getLinkState(env.ref_robot, idx)[0])
        dist = np.linalg.norm(rbt - ref)
        evaluate_buffer_pose['pose_dist_'+str(idx)].append(dist)
        dist_mpjpe += dist
    mpjpe = dist_mpjpe/env.endEff_idx

    return mpjpe

def evaluate():

    evaluate_buffer_angl = {
        'angl_diff_0': [],
        'angl_diff_1': [],
        'angl_diff_2': [],
        'angl_diff_3': [],
        'angl_diff_4': [],
        'angl_diff_5': [],
    }

    evaluate_buffer_pose = {
        'pose_dist_0': [],
        'pose_dist_1': [],
        'pose_dist_2': [],
        'pose_dist_3': [],
        'pose_dist_4': [],
        'pose_dist_5': [],
    }

    evaluate_buffer_edef = {
        'edef_dist': [],
    }
    
    args = ArgumentParser().parse_args()
    if args.task == 'imitate':
        env = robotGymImitateEnv()
        env_name = "robotGymImitateEnv"
    elif args.task == 'push':
        env = robotGymPushEnv()
        env_name = "robotGymPushEnv"

    has_continuous_action_space = True

    max_ref = env.ref_motion.shape[0]           # max timesteps in one episode
    action_std = 0.1                            # set same std for action distribution which was used while saving

    render = False                              # render environment on screen
    frame_delay = 0                             # if required; add delay b/w frames

    total_test_episodes = 1                     # total num of testing episodes

    K_epochs = 80                               # update policy for K epochs
    eps_clip = 0.2                              # clip parameter for PPO
    gamma = 0.99                                # discount factor

    lr_actor = 0.0003                           # learning rate for actor
    lr_critic = 0.001                           # learning rate for critic

    # state space dimension
    state_dim = env.init_obs['observation'].shape[0]

    # action space dimension
    if has_continuous_action_space:
        action_dim = env.action_space.shape[0]
    else:
        action_dim = env.action_space.n

    # initialize a PPO agent
    ppo_agent = PPO(state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, has_continuous_action_space, action_std)

    # preTrained weights directory
    random_seed = 0                 # set this to load a particular checkpoint trained on random seed
    run_num_pretrained = 0          # set this to load a particular checkpoint num

    directory = "ppo_results/ppo_preTrained" + '/' + env_name + '/'
    checkpoint_path = directory + "ppo_{}_{}_{}.pth".format(env_name, random_seed, run_num_pretrained)
    print("loading network from : " + checkpoint_path)
    print("———————————————————————————————————————————————————————————————")

    # load preTrained weights into PPO
    ppo_agent.load(checkpoint_path)

    # reset evaluation values
    eval_running_reward = 0

    # test starts
    for ep in range(1, total_test_episodes+1):

        env.ref_idx = 0
        ep_reward = 0
        ep_m1 = 0
        ep_m2 = 0
        ep_m3 = 0
        state = env.reset()['observation']
        print("episode starts")

        for t in range(1, max_ref+1):
            
            action = ppo_agent.select_action(state)
            state, reward, done, _ = env.step(action)
            
            m1 = similarity_evaluate(env, evaluate_buffer_angl)
            ep_m1 += m1

            m2 = endeff_evaluate(env, evaluate_buffer_edef)
            ep_m2 += m2

            m3 = mpjpe_evaluate(env, evaluate_buffer_pose)
            ep_m3 += m3
            
            reward = reward[0]
            state = state['observation']

            ep_reward += reward

            if render:
                env.render()
                time.sleep(frame_delay)

            if done:
                break

        ppo_agent.buffer.clear()

        ep_m1 = ep_m1 / max_ref
        ep_m2 = ep_m2 / max_ref
        ep_m3 = ep_m3 / max_ref

        eval_running_reward += ep_reward
        print('Episode: {}\t Reward: {} \t Similarity: {} \t Endeff: {} \t MPJPE: {}'.format(ep, round(ep_reward, 2), round(np.sum(ep_m1), 5), round(ep_m2, 5), round(ep_m3, 5)))
        ep_reward = 0

    env.close()

    writer = pd.ExcelWriter('ppo_results/evaluation.xlsx', engine='xlsxwriter')
    pd.DataFrame(evaluate_buffer_angl).to_excel(writer, sheet_name='evaluate_buffer_angl', index=False)
    pd.DataFrame(evaluate_buffer_pose).to_excel(writer, sheet_name='evaluate_buffer_pose', index=False)
    pd.DataFrame(evaluate_buffer_edef).to_excel(writer, sheet_name='evaluate_buffer_edef', index=False)
    writer.close()

    print(args.video)

    avg_test_reward = eval_running_reward / total_test_episodes
    avg_test_reward = round(avg_test_reward, 2)
    print("———————————————————————————————————————————————————————————————")
    print("average test reward : " + str(avg_test_reward))
    print("———————————————————————————————————————————————————————————————")

if __name__ == '__main__':

    evaluate()
