import os
import torch
import numpy as np
import textwrap
from args import ArgumentParser
from robot_env.env_imitate import robotGymImitateEnv
from robot_env.env_push import robotGymPushEnv
from datetime import datetime
from ppo.ppo import PPO

def train(task='imitate'):

    args = ArgumentParser().parse_args()

    has_continuous_action_space = args.has_continuous_action_space

    max_ep_len = args.max_ep_len                            # max timesteps in one episode
    max_training_timesteps = args.max_training_timesteps    # break training loop if timeteps > max_training_timesteps

    print_freq = max_ep_len * 10                            # print avg reward in the interval (in num timesteps)
    log_freq = max_ep_len * 2                               # log avg reward in the interval (in num timesteps)
    save_model_freq = args.save_model_freq                  # save model frequency (in num timesteps)

    action_std = args.action_std                            # starting std for action distribution (Multivariate Normal)
    action_std_decay_rate = args.action_std_decay_rate      # linearly decay action_std (action_std = action_std - action_std_decay_rate)
    min_action_std = args.min_action_std                    # minimum action_std (stop decay after action_std <= min_action_std)
    action_std_decay_freq = args.action_std_decay_freq      # action_std decay frequency (in num timesteps)
    
    # Note : print/log frequencies should be > than max_ep_len

    #———————————————————————————————————————————————————————————————————————————————————————————————————————
    # PPO hyperparameters
    #———————————————————————————————————————————————————————————————————————————————————————————————————————
    
    update_timestep = max_ep_len * 4                        # update policy every n timesteps
    K_epochs = args.K_epochs                                # update policy for K epochs in one PPO update

    eps_clip = args.eps_clip                                # clip parameter for PPO
    gamma = args.gamma                                      # discount factor

    lr_actor = args.lr_actor                                # learning rate for actor network
    lr_critic = args.lr_critic                              # learning rate for critic network

    random_seed = args.random_seed                          # set random seed if required (0 = no random seed)
    
    #———————————————————————————————————————————————————————————————————————————————————————————————————————
    # initialize dimensions
    #———————————————————————————————————————————————————————————————————————————————————————————————————————

    print("initializing imitation training environment")
    if task == 'imitate':
        env = robotGymImitateEnv()
        env_name = "robotGymImitateEnv"
    elif task == 'push':
        env = robotGymPushEnv()
        env_name = "robotGymPushEnv"

    # state space dimension
    state_dim = env.init_obs['observation'].shape[0]
    
    # action space dimension
    if has_continuous_action_space:
        action_dim = env.action_space.shape[0]
    else:
        action_dim = env.action_space.n

    #———————————————————————————————————————————————————————————————————————————————————————————————————————
    # logging paths
    #———————————————————————————————————————————————————————————————————————————————————————————————————————
        
    # log files for multiple runs are NOT overwritten
    log_dir = "ppo_results/ppo_logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    log_dir = log_dir + '/' + env_name + '/'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # get number of log files in log directory
    run_num = 0
    current_num_files = next(os.walk(log_dir))[2]
    run_num = len(current_num_files)

    # create new log file for each run
    log_f_name = log_dir + '/ppo_' + env_name + "_log_" + str(run_num) + ".csv"

    print("current logging run number for " + env_name + " : ", run_num)
    print("logging at : " + log_f_name)

    #———————————————————————————————————————————————————————————————————————————————————————————————————————
    # checkpoint paths
    #———————————————————————————————————————————————————————————————————————————————————————————————————————
    
    run_num_pretrained = 0 # change this to prevent overwriting weights in same env_name folder
    # current_num_files = next(os.walk(log_dir))[2]
    # run_num = len(current_num_files)

    directory = "ppo_results/ppo_preTrained"
    if not os.path.exists(directory):
        os.makedirs(directory)

    directory = directory + '/' + env_name + '/'
    if not os.path.exists(directory):
        os.makedirs(directory)

    run_num_pretrained = 0 # change this to prevent overwriting weights in same env_name folder
    current_num_files = next(os.walk(directory))[2]
    run_num = len(current_num_files)

    checkpoint_path = directory + "ppo_{}_{}_{}.pth".format(env_name, random_seed, run_num_pretrained)
    print("save checkpoint path : " + checkpoint_path)
    
    #———————————————————————————————————————————————————————————————————————————————————————————————————————
    # list all hyperparameters
    #———————————————————————————————————————————————————————————————————————————————————————————————————————

    info_str = """
    ———————————————————————————————————————————————————————————————
    max training timesteps :                        {0}
    max timesteps per episode :                     {1}
    model saving frequency :                        {2}
    log frequency :                                 {3}
    printing average reward over episodes in last : {4}
    ———————————————————————————————————————————————————————————————
    state space dimension :                         {5}
    action space dimension :                        {6}
    ———————————————————————————————————————————————————————————————
    """.format(max_training_timesteps, max_ep_len, save_model_freq, log_freq, print_freq, state_dim, action_dim)
    info_str = textwrap.dedent(info_str)
    print(info_str)

    if has_continuous_action_space:
        info_str = """
        Initializing a continuous action space policy
        ———————————————————————————————————————————————————————————————
        starting std of action distribution :           {0}
        decay rate of std of action distribution :      {1}
        minimum std of action distribution :            {2}
        decay frequency of std of action distribution : {3}
        """.format(action_std, action_std_decay_rate, min_action_std, action_std_decay_freq)
        info_str = textwrap.dedent(info_str)
        print(info_str)

    else:
        print("Initializing a discrete action space policy")
    
    info_str = """
    ———————————————————————————————————————————————————————————————
    PPO update frequency :              {0}
    PPO K epochs :                      {1}
    PPO epsilon clip :                  {2}
    discount factor (gamma) :           {3}
    ———————————————————————————————————————————————————————————————
    optimizer learning rate actor :     {4}
    optimizer learning rate critic :    {5}
    """.format(update_timestep, K_epochs, eps_clip, gamma, lr_actor, lr_critic)
    info_str = textwrap.dedent(info_str)
    print(info_str)

    if random_seed:
        print("setting random seed to ", random_seed)
        torch.manual_seed(random_seed)
        env.seed(random_seed)
        np.random.seed(random_seed)

    #———————————————————————————————————————————————————————————————————————————————————————————————————————
    # training procedure
    #———————————————————————————————————————————————————————————————————————————————————————————————————————

    # initialize a PPO agent
    ppo_agent = PPO(state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, has_continuous_action_space, action_std)

    # track total training time
    start_time = datetime.now().replace(microsecond=0)
    print("Started training at (GMT) : ", start_time)

    # logging file
    log_f = open(log_f_name,"w+")
    log_f.write('episode,timestep,reward\n')

    # printing and logging variables
    print_running_reward = 0
    print_running_episodes = 0

    log_running_reward = 0
    log_running_episodes = 0

    time_step = 0
    i_episode = 0

    # training loop
    while time_step <= max_training_timesteps:

        state = env.reset()['observation']
        current_ep_reward = 0

        for t in range(1, max_ep_len+1):

            # select action with policy
            action = ppo_agent.select_action(state)
            state, reward, done, _ = env.step(action)
            
            reward = reward[0]
            state = state['observation']

            # saving reward and is_terminals
            ppo_agent.buffer.rewards.append(reward)
            ppo_agent.buffer.is_terminals.append(done)

            time_step +=1
            current_ep_reward += reward

            # update PPO agent
            if time_step % update_timestep == 0:
                ppo_agent.update()

            # if continuous action space; then decay action std of ouput action distribution
            if has_continuous_action_space and time_step % action_std_decay_freq == 0:
                ppo_agent.decay_action_std(action_std_decay_rate, min_action_std)

            # log in logging file
            if time_step % log_freq == 0:

                # log average reward till last episode
                log_avg_reward = log_running_reward / log_running_episodes
                log_avg_reward = round(log_avg_reward, 4)

                log_f.write('{},{},{}\n'.format(i_episode, time_step, log_avg_reward))
                log_f.flush()

                log_running_reward = 0
                log_running_episodes = 0

            # printing average reward
            if time_step % print_freq == 0:

                # print average reward till last episode
                print_avg_reward = print_running_reward / print_running_episodes
                print_avg_reward = round(print_avg_reward, 2)

                print("Episode : {} \t\t Timestep : {} \t\t Average Reward : {}".format(i_episode, time_step, print_avg_reward))

                print_running_reward = 0
                print_running_episodes = 0

            # save model weights
            if time_step % save_model_freq == 0:
                info_str = """
                ———————————————————————————————————————————————————————————————
                saving model at :   {0}
                model saved
                Elapsed Time  :     {1}
                ———————————————————————————————————————————————————————————————
                """.format(checkpoint_path, datetime.now().replace(microsecond=0) - start_time)
                info_str = textwrap.dedent(info_str)
                print(info_str)

                ppo_agent.save(checkpoint_path)

            # break; if the episode is over
            if done:
                break

        print_running_reward += current_ep_reward
        print_running_episodes += 1

        log_running_reward += current_ep_reward
        log_running_episodes += 1

        i_episode += 1

    log_f.close()
    env.close()

    # print total training time
    end_time = datetime.now().replace(microsecond=0)
    info_str = """
    ———————————————————————————————————————————————————————————————
    Started training at (GMT) : {0}
    Finished training at (GMT) : {1}
    Total training time  : {2}
    ———————————————————————————————————————————————————————————————
    """.format(start_time, end_time, end_time - start_time)
    info_str = textwrap.dedent(info_str)
    print(info_str)

if __name__ == '__main__':

    train()