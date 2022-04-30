# SeniorProject

## Project Idea:
Training a car in simulation using Deep Reinforcement Learning. I set up a simulation environment in Unreal4 by copying the assets from Carla to build the towns and using the Airsim plugin to have an easy to use API.

## Dependencies:
tensorflow 2 - numpy - matplotlib - gym

## This Repo:
The Duel_DDQN folder is a package for training with Dueling Double DQN using a gym environment. I have modified the code from [here](https://github.com/ChuaCheowHuan/reinforcement_learning/blob/master/DQN_variants/duel_DDQN/duelling_DDQN_cartpole.ipynb) to make it into an easy to use package. Once migrated to tf2, you can train with Dueling Double DQN following the setup template in train.py

## To-Do:
- Migrate code to tf2. The code from the link above only works with tf1 and some function calls are deprecated. 
- Set up custom gym environment for the car in AirSim 
- Design reward function for the car agent that uses the simulation env to determine if the car is in the lane
