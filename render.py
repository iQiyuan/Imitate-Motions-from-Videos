import os
import glob
from tqdm import tqdm
from PIL import Image
from ppo.ppo import PPO
from args import ArgumentParser
from robot_env.env_imitate import robotGymImitateEnv
from robot_env.env_push import robotGymPushEnv

#———————————————————————————————————————————————————————————————————————————————————————————————————————
# save images for gif
#———————————————————————————————————————————————————————————————————————————————————————————————————————


def save_gif_images(env_name, has_continuous_action_space, max_ep_len, action_std):
	
	print("———————————————————————————————————————————————————————————————")

	total_test_episodes = 1     # save gif for only one episode
	K_epochs = 80               # update policy for K epochs
	eps_clip = 0.2              # clip parameter for PPO
	gamma = 0.99                # discount factor

	lr_actor = 0.0003         	# learning rate for actor
	lr_critic = 0.001         	# learning rate for critic

	if env_name == 'robotGymImitateEnv':
		env = robotGymImitateEnv()
	elif env_name == 'robotGymPushEnv':
		env = robotGymPushEnv()

	# state space dimension
	state_dim = env.init_obs['observation'].shape[0]

	# action space dimension
	if has_continuous_action_space:
		action_dim = env.action_space.shape[0]
	else:
		action_dim = env.action_space.n

	# make directory for saving gif images
	gif_images_dir = "ppo_results/ppo_gif_images" + '/'
	if not os.path.exists(gif_images_dir):
		os.makedirs(gif_images_dir)

	# make environment directory for saving gif images
	gif_images_dir = gif_images_dir + '/' + env_name + '/'
	if not os.path.exists(gif_images_dir):
		os.makedirs(gif_images_dir)

	# make directory for gif
	gif_dir = "ppo_results/ppo_gifs" + '/'
	if not os.path.exists(gif_dir):
		os.makedirs(gif_dir)

	# make environment directory for gif
	gif_dir = gif_dir + '/' + env_name  + '/'
	if not os.path.exists(gif_dir):
		os.makedirs(gif_dir)

	ppo_agent = PPO(state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, has_continuous_action_space, action_std)

	# preTrained weights directory
	random_seed = 0             	# set this to load a particular checkpoint trained on random seed
	run_num_pretrained = 0      	# set this to load a particular checkpoint num

	directory = "ppo_results/ppo_preTrained" + '/' + env_name + '/'
	checkpoint_path = directory + "ppo_{}_{}_{}.pth".format(env_name, random_seed, run_num_pretrained)
	print("loading network from : " + checkpoint_path)

	ppo_agent.load(checkpoint_path)

	print("———————————————————————————————————————————————————————————————")

	test_running_reward = 0

	for ep in range(1, total_test_episodes+1):

		ep_reward = 0
		state = env.reset()['observation']

		for t in tqdm(range(1, max_ep_len+1)):

			action = ppo_agent.select_action(state)
			state, reward, done, _ = env.step(action)

			reward = reward[0]
			state = state['observation']

			ep_reward += reward
			
			img = env.render()
			img = Image.fromarray(img.astype('uint8'))
			img.save(gif_images_dir + '/' + str(t) + '.jpg')
			
			if done:
				break

		ppo_agent.buffer.clear()

		test_running_reward +=  ep_reward
		print('Episode: {} \t\t Reward: {}'.format(ep, round(ep_reward, 2)))
		ep_reward = 0

	env.close()

	print("———————————————————————————————————————————————————————————————")
	print("total number of frames / timesteps / images saved : ", t)
	avg_test_reward = test_running_reward / total_test_episodes
	avg_test_reward = round(avg_test_reward, 2)
	print("average test reward : " + str(avg_test_reward))
	print("———————————————————————————————————————————————————————————————")

#———————————————————————————————————————————————————————————————————————————————————————————————————————
# generate gif from saved images
#———————————————————————————————————————————————————————————————————————————————————————————————————————

def save_gif(env_name):

	print("———————————————————————————————————————————————————————————————")

	gif_num = 0     			# change this to prevent overwriting gifs in same env_name folder
	start_timesteps = 500 		# select where to start GIF
	total_timesteps = 800 		# select how many timesteps for GIF frame candidate
	num_frames = 50 			# select how many GIF frames
	step = 5					# select every how many timesteps to save GIF frame
	frame_duration = 10 		# select how many milliseconds each GIF frame lasts

	# input images
	gif_images_dir = "ppo_results/ppo_gif_images/" + env_name + '/*.jpg'

	# ouput gif path
	gif_dir = "ppo_results/ppo_gifs"
	if not os.path.exists(gif_dir):
		os.makedirs(gif_dir)

	gif_dir = gif_dir + '/' + env_name
	if not os.path.exists(gif_dir):
		os.makedirs(gif_dir)

	gif_path = gif_dir + '/ppo_' + env_name + '_gif_' + str(gif_num) + '.gif'

	img_paths = sorted(glob.glob(gif_images_dir))
	img_paths = img_paths[start_timesteps:start_timesteps+total_timesteps]
	selected_indices = [round(i * (total_timesteps - 1) / (num_frames - 1)) for i in range(num_frames)]
	img_paths = [img_paths[i] for i in selected_indices]

	print("total frames in gif : ", len(img_paths))
	print("total duration of gif : " + str(round(len(img_paths) * frame_duration / 1000, 2)) + " seconds")

	# save gif
	img, *imgs = [Image.open(f) for f in img_paths]
	img.save(fp=gif_path, format='GIF', append_images=imgs, save_all=True, optimize=True, duration=frame_duration, loop=0)

	print("saved gif at : ", gif_path)
	print("———————————————————————————————————————————————————————————————")

#———————————————————————————————————————————————————————————————————————————————————————————————————————
# check gif byte size
#———————————————————————————————————————————————————————————————————————————————————————————————————————

def list_gif_size(env_name):
	
	print("———————————————————————————————————————————————————————————————")
	gif_dir = "ppo_results/ppo_gifs/" + env_name + '/*.gif'
	gif_paths = sorted(glob.glob(gif_dir))
	for gif_path in gif_paths:
		file_size = os.path.getsize(gif_path)
		print(gif_path + '\t\t' + str(round(file_size / (1024 * 1024), 2)) + " MB")
	print("———————————————————————————————————————————————————————————————")


if __name__ == '__main__':

	args = ArgumentParser().parse_args()

	if args.task == 'imitate':
		env_name = "robotGymImitateEnv"
	elif args.task == 'push':
		env_name = "robotGymPushEnv"

	has_continuous_action_space = True
	max_ep_len = 2000           # max timesteps in one episode
	action_std = 0.1            # set same std for action distribution which was used while saving

	save_gif_images(env_name, has_continuous_action_space, max_ep_len, action_std)
	save_gif(env_name)
	list_gif_size(env_name)
