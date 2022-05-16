import setup_path
import airsim
import numpy as np
import math
import time

import gym
from gym import spaces
from airgym.envs.airsim_env import AirSimEnv


class AirSimCarEnv(AirSimEnv):
    def __init__(self, ip_address, image_shape):
        super().__init__(image_shape)

        self.image_shape = image_shape
        self.start_ts = 0

        self.state = {
            "position": np.zeros(3),
            "prev_position": np.zeros(3),
            "pose": None,
            "prev_pose": None,
            "collision": False,
        }

        self.car = airsim.CarClient(ip=ip_address)
        self.action_space = spaces.Discrete(6)

        # self.image_request = airsim.ImageRequest(
        #     "0", airsim.ImageType.DepthPerspective, True, False
        # )
        self.image_request = airsim.ImageRequest(
            "1", airsim.ImageType.Scene, False, False
        )

        self.car_controls = airsim.CarControls()
        self.car_state = None

        self.pts = []
        # Down the big road
        for i in range(19):
            self.pts.append(np.array([5*i, 0, 0]))
        # First Turn
        for i in range(10):
            self.pts.append(np.array([97.3, 5*i, 0]))
        # Second Turn
        for i in range(12):
            self.pts.append(np.array([97.3-5*i, 53.5, 0]))
        # Third Turn
        for i in range(10):
            self.pts.append(np.array([35.6, 53.5-5*i, 0]))

    def _setup_car(self):
        self.car.reset()
        self.car.enableApiControl(True)
        self.car.armDisarm(True)
        time.sleep(0.01)

    def __del__(self):
        self.car.reset()

    def _do_action(self, action):
        self.car_controls.brake = 0
        self.car_controls.throttle = 1

        if action == 0: # brake
            self.car_controls.throttle = 0
            self.car_controls.brake = 1
        elif action == 1: # Go straight
            self.car_controls.steering = 0
        elif action == 2: # Go right by a lot
            self.car_controls.steering = 0.5
        elif action == 3: # Go left by a lot
            self.car_controls.steering = -0.5
        elif action == 4: # Go right a little
            self.car_controls.steering = 0.25
        else: # Go left a little
            self.car_controls.steering = -0.25

        self.car.setCarControls(self.car_controls)
        time.sleep(1)

    def transform_obs(self, response):
        # img1d = np.array(response.image_data_float, dtype=np.float)
        # img1d = 255 / np.maximum(np.ones(img1d.size), img1d)
        # img2d = np.reshape(img1d, (response.height, response.width))

        img1d = np.fromstring(response.image_data_uint8, dtype=np.uint8)
        img_rgb = img1d.reshape(response.height, response.width, 3)
        img_rgb = np.flipud(img_rgb)

        from PIL import Image

        image = Image.fromarray(img_rgb)
        image = image.resize((84, 84))
        im_final = np.array(image.convert("L"))

        return im_final.reshape([84, 84, 1])

    def _get_obs(self):
        responses = self.car.simGetImages([self.image_request])
        image = self.transform_obs(responses[0])

        self.car_state = self.car.getCarState()

        self.state["prev_pose"] = self.state["pose"]
        self.state["pose"] = self.car_state.kinematics_estimated
        self.state["collision"] = self.car.simGetCollisionInfo().has_collided

        return image

    def _compute_reward(self):
        MAX_SPEED = 15
        MIN_SPEED = 3
        THRESH_DIST = 4.5
        BETA = 3

        pts = self.pts
        car_pt = self.state["pose"].position.to_numpy_array()

        dist = 10000000
        for i in range(0, len(pts) - 1):
            dist = min(
                dist,
                np.linalg.norm(
                    np.cross((car_pt - pts[i]), (car_pt - pts[i + 1]))
                )
                / np.linalg.norm(pts[i] - pts[i + 1]),
            )

        # print(dist)
        if dist > THRESH_DIST:
            reward = -3
        else:
            reward_dist = math.exp(-BETA * dist) - 0.5
            reward_speed = (
                (self.car_state.speed - MIN_SPEED) / (MAX_SPEED - MIN_SPEED)
            ) - 0.5
            reward = reward_dist + reward_speed

        done = 0
        if reward < -1:
            done = 1
        if self.car_controls.brake == 0:
            if self.car_state.speed <= 1:
                done = 1
            reward = -5
        if self.state["collision"]:
            done = 1
            reward = -10

        return reward, done

    def step(self, action):
        self._do_action(action)
        obs = self._get_obs()
        reward, done = self._compute_reward()

        return obs, reward, done, self.state

    def reset(self):
        self._setup_car()
        self._do_action(1)
        return self._get_obs()
