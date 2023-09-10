import threading
import gym
import highway_env
from matplotlib import pyplot as plt
import pprint
from keyboard import KeyboardEventHandler
import threading
import numpy as np
from datetime import datetime
from highway_env.vehicle.behavior import IDMVehicle
import string
import model
import torch
import utils
from numpy import random
from policyGradient import Policy
from TD3 import TD3
from DDPG import DDPG
import torch.nn.functional as F
import math

f = None

time_steps = 2500
max_action = 0.3
expl_noise = 0.1
batch_size = 256
teraminal_penalty = -100

lat_P = 1
lat_I = 0.15
lat_D = 0
long_P = 1.0
long_I = 1.0
long_D = 1.0
PID_target_time = 0.3
use_keyboard = False

max_iteration = int(1e6)
def main():
    global f

    generate_data = False
    num_of_other_vehicles = 0
    num_of_lanes = 5
    env = gym.make('highway-v0')
    # env.config["show_trajectories"] = True
    env.config["vehicles_count"] = num_of_other_vehicles
    env.config["simulation_frequency"] = 10
    env.config["policy_frequency"] = 10
    env.configure({
        "lanes_count": num_of_lanes,
        "action": {
            "type" : "ContinuousAction"
        },
        # "collision_reward": -100,
        "duration": 600,
        "on_road_reward" : 1,
        # "off_road_reward" : -5,
        'offroad_terminal': True,
        'high_speed_reward': 0.1,
        'screen_height': 600,
        'screen_width': 2400,
        # "observation": {
        #     "type": "Kinematics",
        #     "vehicles_count": num_of_other_vehicles,
        #     "features": ["presence", "x", "y", "vx", "vy", "heading","lat_off"],
        #     "absolute": True,
        #     "normalize": False,
        #     "order": "sorted"
        # }

        # "observation": {
        # "type": "OccupancyGrid",
        # "vehicles_count": num_of_other_vehicles,
        # # "vehicles_count": 15,
        # # "features": ["presence", "x", "y", "vx", "vy", "cos_h", "sin_h"],
        # "features": ["presence", "x", "y", "lat_off", "heading","ang_off","vx", "vy"],
        # "features_range": {
        #     "x": [-100, 100],
        #     "y": [-100, 100],
        #     "vx": [-20, 20],
        #     "vy": [-20, 20]
        # },
        # "grid_size": [[-25, 25], [-25, 25]],
        # "grid_step": [2.5, 2.5],
        # "absolute": False
        # }
        "observation":{
           "type": "GrayscaleObservation",
           "observation_shape": (128, 64),
           "stack_size": 1,
           "weights": [0.2989, 0.5870, 0.1140],  # weights for RGB conversion
           "scaling": 1.75}
        
    })
    if generate_data:
        dir = '/Users/hwpark/Desktop/highway_env/data/'
        date_time = datetime.now()
        dir += date_time.strftime("%Y %m %d %H %M")
        f = open(dir,'w')

    obs = None
    
    policy = DDPG()

    episode_reward = 0
    max_time_step = 10000
    episode_num = 0

    # ego = env.road.vehicles[0].position
    # ego_lane_idx = np.array(env.road.network.get_closest_lane_index(np.array(ego))[2],np.float32)
    state = env.reset()
    for n_epi in range(max_iteration):
        print("\n \n \n")

        # acceleration and steering angle
        done = False
        # ego = env.road.vehicles[0].position
        # ego_heading = env.road.vehicles[0].heading / math.pi
        # ego_speed = [env.road.vehicles[0].speed / 3.6 * math.cos(ego_heading), env.road.vehicles[0].speed / 3.6 * math.sin(ego_heading)]
        # ego_lane_idx = np.array(env.road.network.get_closest_lane_index(np.array(ego))[2],np.float32)
        if use_keyboard:
            if keyboard_listener.is_space_pressed:
                print("wait!!")
            while keyboard_listener.is_space_pressed:
                pass
        
        time_step = 0
        while time_step < max_time_step and not done:
            action = policy.getAction(state)
            print("")
            print("episode num", episode_num)
            print("time step", time_step)
            print("memory size", policy.getMemorySize())
            print("action", action)
            s_prime, reward, done, info = env.step(action)
            policy.insertMemory(state, action, reward, s_prime, done)
            episode_reward += reward
            state = s_prime
            time_step +=1
            env.render()
        
        if policy.isMemoryFull():
            policy.startTraining()
        print("total reward", episode_reward)
        print("getParams", policy.getParams())

        if done or keyboard_listener.reset_flag:
            obs = env.reset()
            keyboard_listener.reset_flag = False
            # +1 to account for 0 indexing. +0 on ep_timesteps since it will increment +1 even if done=True
			# Reset environment 
            state, done = env.reset(), False
            episode_reward = 0
            episode_num += 1
            

        if generate_data:
            f.write(np.array_str(obs))

    # evt.clear()
if __name__ == '__main__':
    evt = threading.Event()
    keyboard_listener = KeyboardEventHandler(evt)
    main()
    if f:
        f.close()

