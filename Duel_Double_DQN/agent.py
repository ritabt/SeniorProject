from importlib import import_module
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
import numpy as np
from Duel_Double_DQN import DuelingDeepQNetwork, ExperienceReplay


class Agent():
    def __init__(self, lr, gamma, n_actions, epsilon, batch_size,
                 input_dims, epsilon_dec=1e-3, eps_end=0.01, eps_update_freq=100,
                 mem_size=100000, fc1_dims=128,
                 fc2_dims=128, replace=100):
        self.action_space = [i for i in range(n_actions)]
        self.gamma = gamma
        self.epsilon = epsilon
        self.eps_dec = epsilon_dec
        self.eps_min = eps_end
        self.eps_update_freq = eps_update_freq
        self.replace = replace
        self.batch_size = batch_size

        self.learn_step_counter = 0
        self.memory = ExperienceReplay(mem_size, input_dims)
        self.q_eval = DuelingDeepQNetwork(n_actions, fc1_dims, fc2_dims)
        self.q_eval.trainable=True
        self.q_next = DuelingDeepQNetwork(n_actions, fc1_dims, fc2_dims)
        self.q_next.trainable=True

        self.q_eval.compile(optimizer=Adam(learning_rate=lr),
                            loss='mean_squared_error')
                            
        self.q_next.compile(optimizer=Adam(learning_rate=lr),
                            loss='mean_squared_error')

    def store_transition(self, state, action, reward, new_state, done):
        self.memory.store_transition(state, action, reward, new_state, done)

    def choose_action(self, observation):
        if np.random.random() < self.epsilon:
            action = np.random.choice(self.action_space)
        else:
            state = np.array([observation])
            actions = self.q_eval.advantage(state)
            action = tf.math.argmax(actions, axis=1).numpy()[0]

        return action

    def update_target(self):
        # TODO: have soft update option
        self.q_next.set_weights(self.q_eval.get_weights()) 

    def update_epsilon(self): 
        if self.learn_step_counter%self.eps_update_freq == 0:
            self.epsilon = max(self.eps_min, self.epsilon * self.eps_dec)

    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return

        if self.learn_step_counter!=0 and self.learn_step_counter % self.replace == 0:
            self.update_target()

        states, actions, rewards, states_, dones = self.memory.sample_buffer(self.batch_size)

        q_pred = self.q_eval(states)
        q_target = q_pred.numpy()
        ##########################################

        # next_state_val = self.target_net.predict(next_states)
        # max_action = np.argmax(self.q_net.predict(next_states), axis=1)
        # batch_index = np.arange(self.batch_size, dtype=np.int32)
        # # q_target = np.copy(target)  #optional  
        # q_target[batch_index, actions] = rewards + self.gamma * next_state_val[batch_index, max_action]*dones

        ##########################################
        q_next = self.q_next(states_)
        q_next = q_next.numpy()
        max_actions = tf.math.argmax(self.q_eval(states_), axis=1)
        batch_index = np.arange(self.batch_size, dtype=np.int32)
        q_target[batch_index, actions] = rewards + self.gamma * q_next[batch_index, max_actions]*(1-dones)
        # for idx, terminal in enumerate(dones):
        #     q_target[idx, actions[idx]] = rewards[idx] + self.gamma*q_next[idx, max_actions[idx]]*(1-int(dones[idx]))

        #########################################
        self.q_eval.train_on_batch(states, q_target)

        self.update_epsilon()

        self.learn_step_counter += 1