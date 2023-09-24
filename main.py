import threading
import gym
import highway_env
from matplotlib import pyplot as plt
import pprint
from keyboard import KeyboardEventHandler
import threading
import numpy as np
from datetime import datetime
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
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import time

import random

random_seed = 42
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed)
np.random.seed(random_seed)
random.seed(random_seed)

f = None


use_keyboard = True

max_iteration = int(1e6)
def clip(action):
    clip_action = [0,0]
    max_clip = 0.1
    for i in range(len(action)):
        if action[i] > max_clip:
            clip_action[i] = max_clip
        elif action[i] < -max_clip:
            clip_action[i] = -max_clip
        else:
            clip_action[i] = action[i]
    return clip_action

def main():
    env = gym.make('Pendulum-v1', max_episode_steps=200, autoreset=True)
    policy = DDPG()
    episode_num = 0
    max_time_step = 200
    print_interval = 20
    score = 0
    countt = 0 
    for n_epi in range(10000):
        state, _ = env.reset()
        done = False
        episode_reward = 0
        time_step = 0

        while time_step < max_time_step and not done:
            action = policy.getAction(state)
            # print("action",action)
            s_prime, reward, done, truncated, info = env.step([action.item()])
            policy.insertMemory(state, action, reward / 100.0, s_prime, done)
            episode_reward += reward
            score += reward
            s = s_prime
            time_step+=1
            # env.render()
        episode_num +=1
        # writer.add_scalar("Q Loss/episode", policy.getQLoss(), episode_num)
        # writer.add_scalar("Mu Loss/episode", policy.getMuLoss(), episode_num)
        writer.add_scalar("episode reward/episode", episode_reward, episode_num)
        
        if policy.getMemorySize() > 2000:
            # print("Memory size", policy.getMemorySize())
            policy.startTraining()
            countt +=1
            # if countt > 1:
                # exit()

        # else:
        #     # print("Memory size", policy.getMemorySize())
        if n_epi%print_interval==0 and n_epi!=0:
            print("# of episode :{}, avg score : {:.1f}".format(n_epi, score/print_interval))
            score = 0.0
# def main():
#     global f

#     generate_data = False
#     num_of_other_vehicles = 0
#     num_of_lanes = 5
#     env = gym.make('racetrack-v0')
#     # env = gym.make('highway-v0')
#     # env.config["show_trajectories"] = True
#     env.config["vehicles_count"] = num_of_other_vehicles
#     env.config["simulation_frequency"] = 10
#     env.config["policy_frequency"] = 10
#     env.configure({
#         "lanes_count": num_of_lanes,
#         "action": {
#             "type" : "ContinuousAction"
#         },
#         # "collision_reward": -100,
#         "duration": 600,
#         "on_road_reward" : 0,
#         # "off_road_reward" : -5,
#         'offroad_terminal': True,
#         # 'high_speed_reward': 0.001,
#         'screen_height': 600,
#         'screen_width': 600,
#         'initial_lane_id': 2,
#         'initial_speed': -1,
#         'other_vehicles': 0,
#         "observation":{
#            "type": "GrayscaleObservation",
#            "observation_shape": (128, 64),
#            "stack_size": 1,
#            "weights": [0.2989, 0.5870, 0.1140],  # weights for RGB conversion
#            "scaling": 1.75}
        
#     })
#     if generate_data:
#         dir = '/Users/hwpark/Desktop/highway_env/data/'
#         date_time = datetime.now()
#         dir += date_time.strftime("%Y %m %d %H %M")
#         f = open(dir,'w')

#     obs = None
    
#     policy = DDPG()

#     episode_reward = 0
#     max_time_step = 10000
#     episode_num = 0
#     max_speed = 0.3
    
#     # ego = env.road.vehicles[0].position
#     # ego_lane_idx = np.array(env.road.network.get_closest_lane_index(np.array(ego))[2],np.float32)
#     state = env.reset()
#     for n_epi in range(max_iteration):
#         print("\n \n \n")

#         # acceleration and steering angle
#         done = False
#         # print("ego lane idx", ego_lane_idx)
#         # if use_keyboard:
#         #     if keyboard_listener.is_space_pressed:
#         #         print("wait!!")
#         #     while keyboard_listener.is_space_pressed:
#         #         pass
        
#         time_step = 0
#         if keyboard_listener.isTrainingMode():
#             print("Training mode")
#         else:
#             print("Evaluation mode")
        
#         is_train_mode = True
#         if use_keyboard and keyboard_listener.isTrainingMode():
#             is_train_mode = True
#         else:
#             is_train_mode = False
#         while time_step < max_time_step and not done:
#             if use_keyboard and keyboard_listener.is_space_pressed:
#                 print("wait!!")
#             while keyboard_listener.is_space_pressed:
#                 pass

#             if is_train_mode:
#                 ego = env.road.vehicles[0].position
#                 ego_heading = env.road.vehicles[0].heading / math.pi
#                 ego_speed = env.road.vehicles[0].speed / 3.6 * math.cos(ego_heading)
#                 action = policy.getAction(state, ego_speed)
#                 # print(action)
#                 clipped_action = clip(action)
                
#                 if abs(ego_speed) > max_speed:
#                     # print(ego_speed,action)
#                     clipped_action[0] = 0
#                 s_prime, reward, done, info = env.step(clipped_action)
                
#                 reward = ego_speed * 0.05
#                 if done:
#                     # print("done!!!!!!!!!!!!!")
#                     reward += teraminal_penalty
#                 policy.insertMemory(state, action, reward, s_prime, done, ego_speed)
#                 episode_reward += reward
#                 state = s_prime

#                 # fig, axes = plt.subplots(ncols=4, figsize=(12, 5))
#                 # fig_state = state
#                 # fig_state = fig_state.reshape(128,64)
#                 # # print("fig size", np.shape(fig_state))
#                 # for i, ax in enumerate(axes.flat):
#                 #     ax.imshow(fig_state, cmap=plt.get_cmap('gray'))
#                 # plt.show()
#             else:
#                 ego = env.road.vehicles[0].position
#                 ego_heading = env.road.vehicles[0].heading / math.pi
#                 ego_speed = env.road.vehicles[0].speed / 3.6 * math.cos(ego_heading)
#                 action = policy.getEvaluationAction(state, ego_speed)
#                 # print(action)
#                 clipped_action = clip(action)
                
#                 if abs(ego_speed) > max_speed:
#                     # print(ego_speed,action)
#                     clipped_action[0] = 0
#                 s_prime, reward, done, info = env.step(clipped_action)
                
#                 reward = ego_speed * 0.05
#                 if done:
#                     reward += teraminal_penalty
#                 episode_reward += reward
#                 state = s_prime

#             time_step +=1
#             env.render()
#         writer.add_scalar("Q Loss/episode", policy.getQLoss(), episode_num)
#         writer.add_scalar("Mu Loss/episode", policy.getMuLoss(), episode_num)
#         writer.add_scalar("episode reward/episode", episode_reward, episode_num)

        
#         if is_train_mode and policy.isMemoryFull():
#             print("Memory size", policy.getMemorySize())
#             policy.startTraining()
#         else:
#             print("Memory size", policy.getMemorySize())
#         print("episode num", episode_num)
#         print("total reward", episode_reward)
#         print("q_loss", policy.getQLoss(), "mu_loss", policy.getMuLoss())
#         # print("getParams", policy.getParams())

#         # if done or keyboard_listener.reset_flag:
#         if done:
#             obs = env.reset()
#             # keyboard_listener.reset_flag = False
#             # +1 to account for 0 indexing. +0 on ep_timesteps since it will increment +1 even if done=True
# 			# Reset environment 
#             state, done = env.reset(), False
#             episode_reward = 0
#             episode_num += 1
            

#         if generate_data:
#             f.write(np.array_str(obs))

    # evt.clear()
if __name__ == '__main__':
    evt = threading.Event()
    keyboard_listener = KeyboardEventHandler(evt)

    now = datetime.now()
    logdir = 'logs/' + now.strftime("%Y%m%d-%H%M%S")
    writer = SummaryWriter(logdir)
    main()
    writer.flush()
    writer.close()
    if f:
        f.close()

