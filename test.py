import time
from robot_env.env_imitate import robotGymImitateEnv
from robot_env.env_push import robotGymPushEnv
from ppo.ppo import PPO
from args import ArgumentParser

def test():
    
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

    total_test_episodes = 10                    # total num of testing episodes

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

    # reset cumulative reward
    test_running_reward = 0

    # test starts
    for ep in range(1, total_test_episodes+1):
        
        ep_reward = 0
        state = env.reset()['observation']

        for t in range(1, max_ref+1):
            
            action = ppo_agent.select_action(state)
            state, reward, done, _ = env.step(action)

            reward = reward[0]
            state = state['observation']

            ep_reward += reward

            if render:
                env.render()
                time.sleep(frame_delay)

            if done:
                break

        ppo_agent.buffer.clear()

        test_running_reward += ep_reward
        print('Episode: {} \t\t Reward: {}'.format(ep, round(ep_reward, 2)))
        ep_reward = 0

    env.close()

    avg_test_reward = test_running_reward / total_test_episodes
    avg_test_reward = round(avg_test_reward, 2)
    print("———————————————————————————————————————————————————————————————")
    print("average test reward : " + str(avg_test_reward))
    print("———————————————————————————————————————————————————————————————")

if __name__ == '__main__':

    test()
