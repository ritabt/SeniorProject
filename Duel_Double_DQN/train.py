import gym
import numpy as np
from Duel_Double_DQN import Agent

# Note: need swig and pip install box2d 
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

if __name__ == '__main__':
    main()
    # env = gym.make('LunarLander-v2')
    # agent = Agent(lr=0.0005, gamma=0.99, n_actions=4, epsilon=1.0,
    #               batch_size=64, input_dims=[8])
    # n_games = 500
    # ddqn_scores = []
    # eps_history = []

    # for i in range(n_games):
    #     done = False
    #     score = 0
    #     observation = env.reset()
    #     while not done:
    #         action = agent.choose_action(observation)
    #         observation_, reward, done, info = env.step(action)
    #         score += reward
    #         agent.store_transition(observation, action, reward, observation_, done)
    #         observation = observation_
    #         agent.learn()
    #     eps_history.append(agent.epsilon)

    #     ddqn_scores.append(score)

    #     avg_score = np.mean(ddqn_scores[-100:])
    #     print('episode: ', i,'score: %.2f' % score,
    #           ' average score %.2f' % avg_score)