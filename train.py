import tensorflow as tf
import gym
import numpy as np
from matplotlib import pyplot as plt

def run_episodes(env, agent, max_episodes):
    r_per_episode = np.array([0])
    cum_R = np.array([0])
    cum_loss = np.array([0])
    cum_R_episodes = 0
    cum_loss_episodes = 0

    # repeat each episode
    for episode_number in range(max_episodes):
        s = env.reset() # reset new episode
        done = False 
        R = 0 
            
        # repeat each step
        while not done:
            # select action using behaviour policy(epsilon-greedy) from model network
            a = agent.act(s)
            # take action in environment
            next_s, r, done, _ = env.step(a)
            # agent learns
            agent.learn(s, a, r, done)
            s = next_s

            R += r 

        (r_per_episode, cum_R_episodes, cum_R, cum_loss_episodes, cum_loss) = stats(r_per_episode, R, cum_R, cum_R_episodes, 
                                                                                agent.cum_loss_per_episode, cum_loss, cum_loss_episodes)
          
    display(r_per_episode, cum_R, cum_loss)

    env.close()


def main():
    env = gym.make('CartPole-v0') # openai gym environment

    max_episodes = 500
    epoch = 100

    num_actions = env.action_space.n # number of possible actions
    obs_size = env.observation_space.shape[0] # dimension of state space
    nhidden = 128 # number of hidden nodes

    epsilon = .9
    gamma = .9
    learning_rate = .3

    replace = 'soft' # params replacement type, 'soft' for soft replacement or empty string '' for hard replacement
    polyak = .001 
    tau_step = 300 

    mem_size = 30000
    minibatch_size = 64

    agent = duel_DDQN_agent(num_actions, obs_size, nhidden,
                        epoch, 
                        epsilon, gamma, learning_rate, 
                        replace, polyak, tau_step,
                        mem_size, minibatch_size)

    run_episodes(env, agent, max_episodes)

if __name__=="__main__":
    main()