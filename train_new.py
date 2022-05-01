import gym
import numpy as np
from Duel_Double_DQN import Agent
import pdb

# Note: need swig and pip install box2d 
def main():
    env = gym.make('CartPole-v1') # openai gym environment

    num_actions = env.action_space.n # number of possible actions
    obs_size = env.observation_space.shape[0] # dimension of state space

    epsilon = 1
    epsilon_dec = 0.95
    gamma = .9
    learning_rate = .3
    minibatch_size = 64

    agent = Agent(lr = learning_rate, gamma=gamma, n_actions=num_actions, 
        epsilon=epsilon, batch_size=minibatch_size, input_dims=[obs_size], epsilon_dec=epsilon_dec)

    n_games = 500
    ddqn_scores = []
    eps_history = []

    for i in range(n_games):
        done = False
        score = 0
        observation = env.reset()
        while not done:
            action = agent.choose_action(observation)
            observation_, reward, done, info = env.step(action)
            score += reward
            agent.store_transition(observation, action, reward, observation_, done)
            observation = observation_
            agent.learn()
        eps_history.append(agent.epsilon)

        ddqn_scores.append(score)

        avg_score = np.mean(ddqn_scores[-100:])
        print('episode: ', i,'score: %.2f' % score,
              ' average score %.2f' % avg_score)

if __name__ == '__main__':
    main()