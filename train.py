import tensorflow as tf
import gym
import numpy as np
from matplotlib import pyplot as plt
from Duel_DDQN import Exp, duel_DDQN_agent, Plot
from time import sleep
import pdb

def get_img(env):
    img = env.render(mode="rgb_array")    
    img_crop = img[125:310, 200:400]
    return np.array([img_crop]).astype(np.float32)

# TODO: move this to Agent class
def get_image_state(model, env):
    img = get_img(env)
    s = model.predict(img)
    s = np.reshape(s, (s.shape[1],))
    # pdb.set_trace()
    # s = s.eval(session=tf.compat.v1.Session())
    return s

def run_episodes(env, agent, max_episodes, plot):
    r_per_episode = np.array([0])
    cum_R = np.array([0])
    cum_loss = np.array([0])
    cum_R_episodes = 0
    cum_loss_episodes = 0
    pretrained_model = agent.preprocess_image_model()
    epsilon = []

    # repeat each episode
    for episode_number in range(max_episodes):
        s = env.reset() # reset new episode
        done = False 
        R = 0 
        # pdb.set_trace()
        if agent.is_conv:
            s = get_image_state(pretrained_model, env)
        # pdb.set_trace()
        # env.render()
        # sleep(0.03)
        # repeat each step
        while not done:

            # select action using behaviour policy(epsilon-greedy) from model network
            a = agent.act(s)
            # take action in environment
            next_s, r, done, _ = env.step(a)
            if agent.is_conv:
                next_s = get_image_state(pretrained_model, env)
            
            # agent learns
            agent.learn(s, a, r, done)
            s = next_s

            R += r 

        (r_per_episode, cum_R_episodes, cum_R, cum_loss_episodes, cum_loss) = plot.stats(r_per_episode, R, cum_R, cum_R_episodes, 
                                                                                agent.cum_loss_per_episode, cum_loss, cum_loss_episodes)
        print('episode: ', episode_number, ' epsilon: %.3f'%agent.epsilon, ' reward: %.2f'%R)
        epsilon.append(agent.epsilon)
    
    epsilon = np.array(epsilon)
    plot.display(r_per_episode, cum_R, cum_loss, max_episodes, epsilon)

    env.close()


def main():
    env = gym.make('CartPole-v0') # openai gym environment

    max_episodes = 10
    epoch = 1000

    num_actions = env.action_space.n # number of possible actions
    obs_size = env.observation_space.shape[0] # dimension of state space
    # pdb.set_trace()
    # obs_size = (185, 200) # img size
    nhidden = 128 # number of hidden nodes

    epsilon = .9
    gamma = .9
    learning_rate = .3

    replace = 'soft' # params replacement type, 'soft' for soft replacement or empty string '' for hard replacement
    polyak = .001 
    tau_step = 300 

    mem_size = 30000
    minibatch_size = 64
    is_conv = True
    img_size = (185, 200, 3)
    if is_conv:
        obs_size = 2048 # size of pretrained model output

    agent = duel_DDQN_agent(num_actions, obs_size, nhidden,
                        epoch, 
                        epsilon, gamma, learning_rate, 
                        replace, polyak, tau_step,
                        mem_size, minibatch_size, is_conv=is_conv, img_size=img_size)
    plot = Plot()

    run_episodes(env, agent, max_episodes, plot)

if __name__=="__main__":
    main()