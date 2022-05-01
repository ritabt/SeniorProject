# SeniorProject

## Project Idea:
Training a car in simulation using Deep Reinforcement Learning. I set up a simulation environment in Unreal4 by copying the assets from Carla to build the towns and using the Airsim plugin to have an easy to use API.

## Dependencies:
Conda environment with python 3.7. Run the following commands to create a working environment for this package (Windows Machine).
```
conda create -n tf python=3.7
conda activate tf
pip install tensorflow==1.15.2
pip install matplotlib
pip install gym
pip install pygame
```

## This Repo:
The Duel_DDQN folder is a package for training with Dueling Double DQN using a gym environment. I have modified the code from [here](https://github.com/ChuaCheowHuan/reinforcement_learning/blob/master/DQN_variants/duel_DDQN/duelling_DDQN_cartpole.ipynb) to make it into an easy to use package. You can train with Dueling Double DQN following the setup template in train.py

## To-Do:
- Add Conv2D layers to preprocess input images
- Modify environment actions to get 2-4 consecutive frames
- Set up custom gym environment for the car in AirSim 
- Design reward function for the car agent that uses the simulation env to determine if the car is in the lane
