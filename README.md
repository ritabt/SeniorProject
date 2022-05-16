# SeniorProject

## Project Idea:
Training a car in simulation using Deep Reinforcement Learning. I set up a simulation environment in Unreal4 by copying the assets from Carla to build the towns and using the Airsim plugin to have an easy to use API. I am using this repo to develop the dueling double DQN part of the project.

## This Repo:
The Duel_DDQN folder is a package for training with Dueling Double DQN using a gym environment. I have modified the code from [here](https://github.com/ChuaCheowHuan/reinforcement_learning/blob/master/DQN_variants/duel_DDQN/duelling_DDQN_cartpole.ipynb) to make it into an easy to use package.

You can train with Dueling Double DQN following the setup template in train.py. 

There are 3 neural network models available: 
- A NN model that can be trained on a state representation (using gym's env.step() or env.reset())
- A NN model that can take an image input and use ResNet50 (cut before the last layer) to get a 2048 vector state representation, then train using the 2048 vector
- A CNN model that has 2 trainable convolutional layers to preprocess the image inputs

train.py uses the cartpole environment from gym to show the dueling double DQN algorithm working. Once you run train.py you can find plots that show progress during training in plots/

## Dependencies:
Conda environment with python 3.7. Run the following commands to create a working environment for this package (Windows Machine).
```
conda create -n tf python=3.7
conda activate tf
pip install tensorflow==1.15.2
pip install tensorflow-gpu==1.15.2
pip install matplotlib
pip install gym
pip install pygame
pip install h5py==2.10.0 --force-reinstall
pip install scikit-image
```
Run the following commands to setup airsim
```
pip install msgpack-rpc-python
pip install airsim
pip install stable-baselines3
```

## To-Do:
- Modify environment actions to get 2-4 consecutive frames as a state
- Set up custom gym environment for the car in AirSim with similar specs
- Design reward function for the car agent that uses the simulation env to determine if the car is in the lane
- Compare performance of different Dueling Double DQN approach with cartpole: using gym's state representation - using ResNet50 as a preprocessing step - training conv2d layers within the DQN.
- Implement Prioritized Experience Replay

