import gym
import random
import collections
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from stateRepresenter import StateRepresenter
from torchsummary import summary
import time
import copy

#Hyperparameters
lr_mu           = 3e-4
lr_q            = 3e-4
lr_s            = 3e-4
gamma           = 0.99
batch_size      = 32
buffer_limit    = 100000
tau             = 0.01 # for target network soft update
number_of_train = 30
policy_noise = 0.02
noise_clip = 0.04
policy_freq=2


# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Implementation of Twin Delayed Deep Deterministic Policy Gradients (TD3)
# Paper: https://arxiv.org/abs/1802.09477

class ReplayBuffer():
	def __init__(self):
		self.buffer = collections.deque(maxlen=buffer_limit)

	def put(self, transition):
		self.buffer.append(transition)
	def sample(self, n):
		mini_batch = random.sample(self.buffer, n)
		# s_lst, a_lst, r_lst, s_prime_lst, done_mask_lst = [], [], [], [], []
		s_lst, a_lst, r_lst, s_prime_lst, done_mask_lst, ego_speed_lst, ego_speed_prime_lst = [], [], [], [], [], [], []
		

		for transition in mini_batch:
			s, a, r, s_prime, done, ego_speed, ego_speed_prime = transition
			# s, a, r, s_prime, done = transition
			s_lst.append(s)
			a_lst.append([a])
			r_lst.append([r])
			s_prime_lst.append(s_prime)
			done_mask = 0.0 if done else 1.0 
			done_mask_lst.append([done_mask])
			ego_speed_lst.append(ego_speed)
			ego_speed_prime_lst.append(ego_speed_prime)
		
		return torch.tensor(s_lst, dtype=torch.float), torch.tensor(a_lst, dtype=torch.float), \
		        torch.tensor(r_lst, dtype=torch.float), torch.tensor(s_prime_lst, dtype=torch.float), \
		        torch.tensor(done_mask_lst, dtype=torch.float), torch.tensor(ego_speed_lst, dtype=torch.float), \
				torch.tensor(ego_speed_prime_lst, dtype=torch.float)

		# return torch.tensor(s_lst, dtype=torch.float), torch.tensor(a_lst, dtype=torch.float), \
		# torch.tensor(r_lst, dtype=torch.float), torch.tensor(s_prime_lst, dtype=torch.float), \
		# torch.tensor(done_mask_lst, dtype=torch.float)
	
	def size(self):
		return len(self.buffer)

class Actor(nn.Module):
	def __init__(self, state_dim, action_dim, max_action):
		super(Actor, self).__init__()

		self.l1 = nn.Linear(state_dim, 256)
		# self.l2 = nn.Linear(256, 256)
		self.l3 = nn.Linear(256, action_dim)
		
		self.max_action = max_action

	def newActFunc5(self, x):
		# grad1 = 2 * 10e-3
		# grad2 = 1 * 10e-4
		grad1 = 4.5 * 10e-2
		grad2 = 3 * 10e-2
		clip = 0.2
		# higher grad out side of the clip
		return torch.where(torch.abs(x) < clip / grad2, grad2 * x, grad1 * x - clip * (grad1 / grad2 - 1))
		# lower grad out side of the clip
		# return torch.where(torch.abs(x) < clip / grad1, grad1 * x, grad2 * x - clip * (grad2 / grad1 - 1))
		

	def forward(self, state):
		# print(state.size())
		a = F.relu(self.l1(state))
		# a = F.relu(self.l2(a))
		# return self.newActFunc5(self.l3(a))
		return F.tanh(self.l3(a)) * self.max_action

class OrnsteinUhlenbeckNoise:
    def __init__(self, mu):
        self.theta, self.dt, self.sigma = 0.1, 0.01, 0.1
        self.mu = mu
        self.x_prev = np.zeros_like(self.mu)

    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + \
                self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x

class Critic(nn.Module):
	def __init__(self, state_dim, action_dim):
		super(Critic, self).__init__()

		# Q1 architecture
		self.l1 = nn.Linear(state_dim + action_dim, 256)
		# self.l2 = nn.Linear(256, 256)
		self.l3 = nn.Linear(256, 1)

		# Q2 architecture
		self.l4 = nn.Linear(state_dim + action_dim, 256)
		# self.l5 = nn.Linear(256, 256)
		self.l6 = nn.Linear(256, 1)


	def forward(self, state, action):
		sa = torch.cat([state, action], 1)

		q1 = F.relu(self.l1(sa))
		# q1 = F.relu(self.l2(q1))
		q1 = self.l3(q1)

		q2 = F.relu(self.l4(sa))
		# q2 = F.relu(self.l5(q2))
		q2 = self.l6(q2)
		return q1, q2


	def Q1(self, state, action):
		sa = torch.cat([state, action], 1)

		q1 = F.relu(self.l1(sa))
		# q1 = F.relu(self.l2(q1))
		q1 = self.l3(q1)
		return q1


class TD3(object):
	def __init__(self, action_dim, max_action):
		state_dim = 12 + 1
		self.memory = ReplayBuffer()
		self.actor = Actor(state_dim, action_dim, max_action)
		self.actor_target = copy.deepcopy(self.actor)
		self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr_mu)

		self.critic = Critic(state_dim, action_dim)
		self.critic_target = copy.deepcopy(self.critic)
		self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr_q)

		self.state_representer = StateRepresenter()
		self.state_optimizer = optim.Adam(self.state_representer.parameters(), lr=lr_s)

		self.actor_loss = 0
		self.critic_loss = 0

		self.max_action = max_action
		self.policy_noise = policy_noise
		self.noise_clip = noise_clip
		self.policy_freq = 2

		self.total_it = 0

		self.ou_noise_accel = OrnsteinUhlenbeckNoise(mu=np.zeros(1))
		self.ou_noise_steer = OrnsteinUhlenbeckNoise(mu=np.zeros(1))

		self.noise_ratio = 0.3
		self.noise_decay_ratio = 1 - 3 * 10e-5
		self.noise_ths = 0.02


	def select_action(self, state, ego_speed):
		state = torch.cat([self.state_representer(torch.from_numpy(state).float()).reshape(12,1), torch.tensor(ego_speed, dtype = torch.float).reshape(1,1)]).reshape(1,13)

		act = self.actor(state).flatten()
		if(self.noise_ratio > self.noise_ths):
			self.noise_ratio *= self.noise_decay_ratio
		return [act[0].item() + self.noise_ratio * self.ou_noise_accel()[0], \
				act[1].item() + self.noise_ratio * self.ou_noise_steer()[0]]

	def train(self):
		for _ in range(number_of_train):
			self.total_it += 1
			# Sample replay buffer 
			state, action, reward, s_prime, not_done, ego_speed, ego_speed_prime = self.memory.sample(batch_size)
			represented_state = self.state_representer(state)
			represented_state = torch.cat([represented_state.reshape(12,batch_size), torch.tensor(ego_speed, dtype = torch.float).reshape(1,batch_size)]).reshape(batch_size,13)
			represented_state_prime = self.state_representer(s_prime)
			represented_state_prime = torch.cat([represented_state_prime.reshape(12,batch_size), torch.tensor(ego_speed_prime, dtype = torch.float).reshape(1,batch_size)]).reshape(batch_size,13)

			action = action.reshape(batch_size,2)
			with torch.no_grad():
				# Select action according to policy and add clipped noise
				noise = (
					torch.randn_like(action) * self.policy_noise
				).clamp(-self.noise_clip, self.noise_clip)
				next_action = (
					self.actor_target(represented_state_prime) + noise
				)

				# Compute the target Q value
				target_Q1, target_Q2 = self.critic_target(represented_state_prime, next_action)
				target_Q = torch.min(target_Q1, target_Q2)
				target_Q = reward + not_done * gamma * target_Q

			# Get current Q estimates
			current_Q1, current_Q2 = self.critic(represented_state, action)

			# Compute critic loss
			self.critic_loss = (F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)) / 100000.0

			# Optimize the critic
			self.critic_optimizer.zero_grad()
			self.state_optimizer.zero_grad()
			self.critic_loss.backward()
			# self.state_optimizer.step()
			self.critic_optimizer.step()

			# Delayed policy updates
			if self.total_it % self.policy_freq == 0:
				# Reinitialize the represented state
				represented_state = self.state_representer(state)
				represented_state = torch.cat([represented_state.reshape(12,batch_size), torch.tensor(ego_speed, dtype = torch.float).reshape(1,batch_size)]).reshape(batch_size,13)
				represented_state_prime = self.state_representer(s_prime)
				represented_state_prime = torch.cat([represented_state_prime.reshape(12,batch_size), torch.tensor(ego_speed_prime, dtype = torch.float).reshape(1,batch_size)]).reshape(batch_size,13)

				# Compute actor loss
				self.actor_loss = -self.critic.Q1(represented_state, self.actor(represented_state)).mean() / 100.0
				
				# Optimize the actor 
				self.actor_optimizer.zero_grad()
				self.state_optimizer.zero_grad()
				self.actor_loss.backward()
				self.state_optimizer.step()
				self.actor_optimizer.step()

				# Update the frozen target models
				for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
					target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

				for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
					target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)


	def getParams(self):
		# print("q")
		# for param in self.critic.parameters():
		# 	print(param.grad)
		# # print("q target")
		# # for param in self.q_target.parameters():
		# #     print(param)
		# print("mu")
		# for param in self.actor.parameters():
		# 	print(param.grad)
		# # print("mu target")
		# # for param in self.mu_target.parameters():
		# #     print(param)
		print("state")
		for param in self.state_representer.parameters():
			print(param)    
			# print("==========================================")
			print(param.grad)
		print("\n")
		# self.q, self.q_target = QNet(), QNet()
		# self.mu, self.mu_target = MuNet(), MuNet()



	def save(self, filename):
		torch.save(self.critic.state_dict(), filename + "_critic")
		torch.save(self.critic_optimizer.state_dict(), filename + "_critic_optimizer")
		
		torch.save(self.actor.state_dict(), filename + "_actor")
		torch.save(self.actor_optimizer.state_dict(), filename + "_actor_optimizer")


	def load(self, filename):
		self.critic.load_state_dict(torch.load(filename + "_critic"))
		self.critic_optimizer.load_state_dict(torch.load(filename + "_critic_optimizer"))
		self.critic_target = copy.deepcopy(self.critic)

		self.actor.load_state_dict(torch.load(filename + "_actor"))
		self.actor_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer"))
		self.actor_target = copy.deepcopy(self.actor)
	
	def getEvaluationAction(self, state, ego_speed):
		a = self.mu.getAction(torch.cat([self.state_representer(torch.from_numpy(state).float()).reshape(12,1), torch.tensor(ego_speed, dtype = torch.float).reshape(1,1)]))
		action = []
		print("action",a[0][0].item(), a[0][1].item())

		action.append(a[0][0].item())
		action.append(a[0][1].item())

		return action
	
	def insertMemory(self, state, action, reward, s_prime, done, ego_speed, ego_speed_prime):
		self.memory.put((state, action ,reward / 10e-3, s_prime, done, ego_speed, ego_speed_prime))

	def isMemoryFull(self):
		return self.memory.size() >= buffer_limit

	def isReadyForTraining(self):
		return self.memory.size() >= buffer_limit

	def getMemorySize(self):
		return self.memory.size()

	def getCriticLoss(self):
		return self.critic_loss
	def getActorLoss(self):
		return self.actor_loss