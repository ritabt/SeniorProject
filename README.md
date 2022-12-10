# SeniorProject

## Project:
Training a car in simulation using Deep Reinforcement Learning. I set up a simulation environment in Unreal4 by copying the assets from Carla to build the towns and using the Airsim plugin to have an easy to use API. I am using this repo to develop the dueling double DQN part of the project.

For more info on the project, see the [final paper](https://github.com/ritabt/SeniorProject_DDDQN/tree/main/final_paper).

## This Repo:
The Duel_DDQN folder is a package for training with Dueling Double DQN using a gym environment. I have modified the code from [here](https://github.com/ChuaCheowHuan/reinforcement_learning/blob/master/DQN_variants/duel_DDQN/duelling_DDQN_cartpole.ipynb) to make it into an easy to use package.

You can train with Dueling Double DQN following the setup template in `train.py` and `train_airsim.py`. 

There are 4 neural network models available: 
- A simple NN model that can take an image input and uses ResNet50 (cut before the last layer and freeze ResNet50 weights) to get a 2048 vector state representation, then train using the 2048 vector
- A larger NN model that can be trained using the same transfer learning approach as the previous NN (ResNet50 preprocessing step for RGB images)
- A CNN model that has 2 trainable convolutional layers to preprocess the image inputs
- A larger CNN model 

`train.py` uses the cartpole environment from gym to show the dueling double DQN algorithm working. Once you run `train.py` you can find plots that show progress during training in plots/

`train_airsim.py` uses the Unreal 4 environment to train a simulated autonomous vehicle. Once you run `train_airsim.py` you can find plots that show progress during training in `plots/`

## Using the Training file

To use this repo with Airsim, put this repo in the `PythonClient/` folder that comes with the Airsim example code. Once you launch the Unreal environment and press Play, you can run `train_airsim.py`.

`train.py` was used for development purposes on Gym's catpole environment and is not kept up to date.

Modify `train_airsim.py` to define learning rates to be used and what neural networks to train on. The neural network are indexed from 0 to 4. If using a CNN set `is_conv=True` to set up the buffer to store images. The reward per episode is printed in the terminal as well as the epsilon. The code automatically plots the reward per episode with the epsilon. Onbserving the smoothed reward per episode increasing as epsilon decreases shows that the agent is learning how to navigate the environment.

## Note about Waypoints:

You must set up waypoints for the reward function to work. Set up `self.pts` in the `AirSimCarEnv` class in `airgym/envs/car_env.py` to be points in the form of (x,y,z) along the middle of the road that you want to car to learn how to drive on. Modify the reward function if you would like to not use the waypoints. However, the training will take longer to converge and the car's behavior will not be optimal without the waypoints.

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


