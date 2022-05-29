import setup_path
import gym
import airgym
import numpy as np
import os
import tensorflow as tf
from matplotlib import pyplot as plt
from Duel_DDQN import Exp, duel_DDQN_agent, Plot
from time import sleep
import pdb
import time
import datetime
from skimage import color
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage



# def get_img(env):
# 	img = env.render(mode="rgb_array")
# 	img = img[125:310, 200:400]
# 	# img = color.rgb2gray(img)
# 	return np.array([img]).astype(np.float32)


# TODO: move this to Agent class
def get_image_state(model, img):
	# img = get_img(env)
	img = np.array([img]).astype(np.float32)
	s = model.predict(img)
	s = np.reshape(s, (s.shape[1],))
	return s

def run_episodes(env, agent, max_episodes, plot, load=False, test=False):
	r_per_episode = np.array([0])
	cum_R = np.array([0])
	cum_loss = np.array([0])
	cum_R_episodes = 0
	cum_loss_episodes = 0
	if not agent.is_conv:
		pretrained_model = agent.preprocess_image_model()
	epsilon = []

	# repeat each episode
	for episode_number in range(max_episodes):
		if episode_number % 50 == 0:
			agent.save_checkpoint()

		s = env.reset()  # reset new episode
		done = False
		R = 0
		# pdb.set_trace()
		if not test:
			if not agent.is_conv:
				s = get_image_state(pretrained_model, s)

		# repeat each step
		while not done:

			# select action using behaviour policy(epsilon-greedy) from model network
			a = agent.act(s)
			# take action in environment
			next_s, r, done, _ = env.step(a)
			if not test:
				if not agent.is_conv:
					next_s = get_image_state(pretrained_model, next_s)

			# agent learns
			agent.learn(s, a, r, done)
			s = next_s

			R += r

		(
			r_per_episode,
			cum_R_episodes,
			cum_R,
			cum_loss_episodes,
			cum_loss,
		) = plot.stats(
			r_per_episode,
			R,
			cum_R,
			cum_R_episodes,
			agent.cum_loss_per_episode,
			cum_loss,
			cum_loss_episodes,
		)
		print(
			"episode: ",
			episode_number,
			" epsilon: %.3f" % agent.epsilon,
			" reward: %.2f" % R,
		)
		epsilon.append(agent.epsilon)

	epsilon = np.array(epsilon)
	plot.display(r_per_episode, cum_R, cum_loss, max_episodes, epsilon, agent.learning_rate, agent.network_idx)

	


def main():
	env = gym.make(
					"airgym:airsim-car-sample-v0",
					ip_address="127.0.0.1",
					image_shape=(144, 256, 3),
			)


	test = False

	max_episodes = 1000
	load = False
	epoch = 100

	num_actions = env.action_space.n  # number of possible actions
	obs_size = env.image_shape
	nhidden = 128  # number of hidden nodes

	epsilon = 0.9
	if load:
		# No exploration
		epsilon = 0.01
	gamma = 0.9

	replace = "soft"  # params replacement type, 'soft' for soft replacement or empty string '' for hard replacement
	polyak = 0.001
	tau_step = 300

	mem_size = 10000
	minibatch_size = 64
	
	obs_size = 2048  # size of pretrained model output

	if test:
		is_conv = False
		img_size = None
		obs_size = env.observation_space.shape[0]

	lrs = [1e-2, 1e-3, 1e-4, 1e-5, 1e-6]
	# lrs = [1e-5]
	if test:
		lrs = [.3]

	neural_networks = [0,1,2,3]
	conv = [False, False, True, True]
	best_lrs = [0.0001, 0.001, 0.0001, 1e-5]

	if load:
		max_episodes = 10
		for i, is_conv, learning_rate in zip(neural_networks, conv, best_lrs):
			if is_conv:
				img_size = env.image_shape
			else:
				img_size = None

			agent = duel_DDQN_agent(
				num_actions,
				obs_size,
				nhidden,
				epoch,
				epsilon,
				gamma,
				learning_rate,
				replace,
				polyak,
				tau_step,
				mem_size,
				minibatch_size,
				is_conv=is_conv,
				img_size=img_size,
				test=test,
				network_idx=i
			)
			agent.load_checkpoint()
			plot = Plot()

			time_1 = time.time()
			run_episodes(env, agent, max_episodes, plot, load=load)
			time_2 = time.time()
			time_interval = time_2 - time_1
			time_taken = str(datetime.timedelta(seconds=time_interval))
			print("Time taken: ", time_taken)
			agent.save_checkpoint()

	else:
		# for i, is_conv in zip(neural_networks, conv):
		for i, is_conv, learning_rate in zip(neural_networks, conv, best_lrs):

			if is_conv:
				img_size = env.image_shape
			else:
				img_size = None

			# for learning_rate in lrs:
			agent = duel_DDQN_agent(
				num_actions,
				obs_size,
				nhidden,
				epoch,
				epsilon,
				gamma,
				learning_rate,
				replace,
				polyak,
				tau_step,
				mem_size,
				minibatch_size,
				is_conv=is_conv,
				img_size=img_size,
				test=test,
				network_idx=i
			)

			plot = Plot()

			time_1 = time.time()
			run_episodes(env, agent, max_episodes, plot, load=load)
			time_2 = time.time()
			time_interval = time_2 - time_1
			time_taken = str(datetime.timedelta(seconds=time_interval))
			print("Time taken: ", time_taken)
			agent.save_checkpoint()


if __name__ == "__main__":
	main()