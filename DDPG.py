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
from torch.optim.lr_scheduler import _LRScheduler

#Hyperparameters
lr_mu           = 0.01
lr_q            = 0.01
lr_s            = 0.01
gamma           = 0.99
batch_size      = 64
buffer_limit    = 30000
tau             = 0.05 # for target network soft update
number_of_train = 30

class ReplayBuffer():
    def __init__(self):
        self.buffer = collections.deque(maxlen=buffer_limit)

    def put(self, transition):
        self.buffer.append(transition)
    
    def sample(self, n):
        mini_batch = random.sample(self.buffer, n)
        s_lst, a_lst, r_lst, s_prime_lst, done_mask_lst, ego_speed_lst, ego_speed_prime_lst = [], [], [], [], [], [], []

        for transition in mini_batch:
            s, a, r, s_prime, done, ego_speed, ego_speed_prime = transition
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
    
    def size(self):
        return len(self.buffer)

class MuNet(nn.Module):
    def __init__(self):
        super(MuNet, self).__init__()
        self.fc1 = nn.Linear(13, 128)
        nn.init.kaiming_normal_(self.fc1.weight, nonlinearity='relu')
        self.bn1 = torch.nn.BatchNorm1d(128)

        self.fc2 = nn.Linear(128, 64)
        nn.init.kaiming_normal_(self.fc2.weight, nonlinearity='relu')
        self.bn2 = torch.nn.BatchNorm1d(64)

        self.fc3 = nn.Linear(64, 2)
        nn.init.kaiming_normal_(self.fc3.weight, nonlinearity='relu')
        self.clipping = torch.tensor([0.2, 0.2])
        self.ELU = torch.nn.ELU()
    
    def newActFunc(self, x):
        # print(x)
        a = torch.exp(10e-4 * x) - 1
        b = -torch.exp(-10e-4 * x) + 1
        # print(a,b)
        # print(torch.where(torch.abs(a) > torch.abs(b), a, b))
        return torch.where(torch.abs(a) > torch.abs(b), a, b)

    def newActFunc2(self, x):
        a = torch.exp(x - 1)
        b = -torch.exp(-x + 1)
        out_of_range = torch.where(torch.abs(a) > torch.abs(b), a, b)
        return torch.where(1 < torch.abs(out_of_range), x, out_of_range)

    def newActFunc3(self, x):
        grad = 3
        a = grad * torch.exp(x - 1)
        b = -grad * torch.exp(-x + 1)
        out_of_range = torch.where(torch.abs(a) > torch.abs(b), a, b)
        return torch.where(grad < torch.abs(out_of_range), x, out_of_range)
    
    def newActFunc4(self, x):
        grad = 10e-5
        clip = 0.2
        # grad = 1
        # clip = 1
        a = grad * torch.exp(x - clip) + grad * (clip - 1)
        b = -grad * torch.exp(-x - clip) - grad * (clip - 1)
        out_of_range = torch.where(x > 0, a, b)
        return torch.where(abs(x) < clip, grad * x, out_of_range)

    def newActFunc5(self, x):
        # grad1 = 2 * 10e-3
        # grad2 = 1 * 10e-4
        grad1 = 1.5 * 10e-3
        grad2 = 8 * 10e-4
        clip = 0.1
        # higher grad out side of the clip
        return torch.where(torch.abs(x) < clip / grad2, grad2 * x, grad1 * x - clip * (grad1 / grad2 - 1))
        # lower grad out side of the clip
        # return torch.where(torch.abs(x) < clip / grad1, grad1 * x, grad2 * x - clip * (grad2 / grad1 - 1))

    def forward(self, x):
        mu = self.fc1(x)
        mu = self.bn1(mu)
        mu = F.relu(mu)
        mu = self.fc2(mu)
        mu = self.bn2(mu)
        mu = F.relu(mu)
        mu = self.fc3(mu)
        mu = F.tanh(mu) * 0.1
        # mu = self.newActFunc5(mu)
        return mu

    def getAction(self, x):
        x = x.reshape(1,13)
        mu = self.fc1(x)
        mu = F.relu(mu)
        mu = self.fc2(mu)
        mu = F.relu(mu)
        mu = self.fc3(mu)
        mu = F.tanh(mu) * 0.1
        # mu = self.newActFunc5(mu)
        return mu

class QNet(nn.Module):
    def __init__(self):
        super(QNet, self).__init__()
        self.fc_q1 = nn.Linear(15, 128)
        nn.init.kaiming_normal_(self.fc_q1.weight, nonlinearity='relu')
        self.bn1 = torch.nn.BatchNorm1d(128)

        self.fc_q2 = nn.Linear(128, 64)
        nn.init.kaiming_normal_(self.fc_q2.weight, nonlinearity='relu')
        self.bn2 = torch.nn.BatchNorm1d(64)

        self.fc_q3 = nn.Linear(64, 1)
        nn.init.kaiming_normal_(self.fc_q3.weight, nonlinearity='relu')
        # self.fc_out = nn.Linear(32,1)

    def forward(self, x, a):
        h1 = x.reshape(batch_size,13)
        h2 = a.reshape(batch_size,2)
        cat = torch.cat([h1,h2], dim=1)
        q = self.fc_q2(self.bn1(self.fc_q1(cat)))
        q = self.fc_q3(self.bn2(q))

        return q

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

class DDPG():
    def __init__(self):
        self.memory = ReplayBuffer()
        self.q, self.q_target = QNet(), QNet()
        self.q_target.load_state_dict(self.q.state_dict())
        self.mu, self.mu_target = MuNet(), MuNet()
        self.mu_target.load_state_dict(self.mu.state_dict())
        self.score = 0.0

        self.mu_optimizer = optim.Adam(self.mu.parameters(), lr=lr_mu)
        self.q_optimizer  = optim.Adam(self.q.parameters(), lr=lr_q)
        
        self.ou_noise_accel = OrnsteinUhlenbeckNoise(mu=np.zeros(1))
        self.ou_noise_steer = OrnsteinUhlenbeckNoise(mu=np.zeros(1))

        self.state_representer = StateRepresenter()
        self.state_optimizer = optim.Adam(self.state_representer.parameters(), lr=lr_s)

        self.q_loss = 0
        self.mu_loss = 0

        self.steer_noise_ratio = 0.1
        self.accel_noise_ratio = 0.1
        self.noise_decay_ratio = 1
        self.noise_ths = 0.02

        self.scheduler_mu = optim.lr_scheduler.LambdaLR(self.mu_optimizer, lr_lambda = lambda epoch: 0.9999 ** epoch)
        self.scheduler_q = optim.lr_scheduler.LambdaLR(self.q_optimizer, lr_lambda = lambda epoch: 0.9999 ** epoch)
        self.scheduler_state = optim.lr_scheduler.LambdaLR(self.state_optimizer, lr_lambda = lambda epoch: 0.9999 ** epoch)
        

    def train(self):
        s,a,r,s_prime,done_mask,ego_speed,ego_speed_prime  = self.memory.sample(batch_size)

        represented_state = self.state_representer(s)
        represented_state = torch.cat([represented_state.reshape(12,batch_size), torch.tensor(ego_speed, dtype = torch.float).reshape(1,batch_size)]).reshape(batch_size,13)
        represented_state_prime = self.state_representer(s_prime)
        represented_state_prime = torch.cat([represented_state_prime.reshape(12,batch_size), torch.tensor(ego_speed_prime, dtype = torch.float).reshape(1,batch_size)]).reshape(batch_size,13)
        
        target = r + gamma * self.q_target(represented_state_prime, self.mu_target(represented_state_prime)) * done_mask
        self.q_loss = F.smooth_l1_loss(self.q(represented_state,a), target.detach())
        self.q_optimizer.zero_grad()
        self.state_optimizer.zero_grad()
        self.q_loss.backward()
        self.q_optimizer.step()
        self.state_optimizer.step()

        represented_state = self.state_representer(s)
        represented_state = torch.cat([represented_state.reshape(12,batch_size), torch.tensor(ego_speed, dtype = torch.float).reshape(1,batch_size)]).reshape(batch_size,13)
        represented_state_prime = self.state_representer(s_prime)
        represented_state_prime = torch.cat([represented_state_prime.reshape(12,batch_size), torch.tensor(ego_speed_prime, dtype = torch.float).reshape(1,batch_size)]).reshape(batch_size,13)

        self.mu_loss = -self.q(represented_state, self.mu(represented_state)).mean() #* 10e9
        self.mu_optimizer.zero_grad()
        self.state_optimizer.zero_grad()
        self.mu_loss.backward()
        self.mu_optimizer.step()
        self.state_optimizer.step()
       

        
    def soft_update(self, net, net_target):
        for param_target, param in zip(net_target.parameters(), net.parameters()):
            param_target.data.copy_(param_target.data * (1.0 - tau) + param.data * tau)
    
    def getAction(self, state, ego_speed):
        # if not self.isMemoryFull():
        #     return [self.noise_ratio * self.ou_noise_accel()[0], self.noise_ratio * self.ou_noise_steer()[0]]
        # a = self.mu.getAction(torch.cat([self.state_representer(torch.from_numpy(state).float()).reshape(12,1), torch.tensor(ego_speed, dtype = torch.float).reshape(1,1)]))
        # action = []
        # print("action",a[0][0].item(), a[0][1].item(), self.noise_ratio * self.ou_noise_accel()[0], self.noise_ratio * self.ou_noise_steer()[0])
        
        # action.append(a[0][0].item() + self.noise_ratio * self.ou_noise_accel()[0])
        # action.append(a[0][1].item() + self.noise_ratio * self.ou_noise_steer()[0])

        # return action

        a = self.mu.getAction(torch.cat([self.state_representer(torch.from_numpy(state).float()).reshape(12,1), torch.tensor(ego_speed, dtype = torch.float).reshape(1,1)]))
        if not self.isMemoryFull():
            # print([self.noise_ratio * self.ou_noise_accel()[0]])
            return [a[0][0].item() + self.accel_noise_ratio * self.ou_noise_accel()[0], a[0][1].item() + self.steer_noise_ratio * self.ou_noise_steer()[0]]
        
        action = []
        # print("action",a[0][0].item(), self.noise_ratio * self.ou_noise_accel()[0], )
        
        action.append(a[0][0].item() + self.accel_noise_ratio * self.ou_noise_accel()[0])
        action.append(a[0][1].item() + self.steer_noise_ratio * self.ou_noise_steer()[0])

        return action
    
    def getEvaluationAction(self, state, ego_speed):
        a = self.mu.getAction(torch.cat([self.state_representer(torch.from_numpy(state).float()).reshape(12,1), torch.tensor(ego_speed, dtype = torch.float).reshape(1,1)]))
        action = []
        # print("action",a[0][0].item(), a[0][1].item())
        
        action.append(a[0][0].item())
        action.append(a[0][1].item())

        return action
    
    def insertMemory(self, state, action, reward, s_prime, done, ego_speed, ego_speed_prime):
        self.memory.put((state, action ,reward / 10e3, s_prime, done, ego_speed, ego_speed_prime))
    
    def isMemoryFull(self):
        return self.memory.size() >= buffer_limit
    
    def getMemory(self):
        return self.memory
    
    def startTraining(self):
        print("Start Training !")
        print("accel noise", self.accel_noise_ratio * self.ou_noise_accel()[0], "steer", self.steer_noise_ratio * self.ou_noise_steer()[0])
        for i in range(number_of_train):
            self.train()
            self.soft_update(self.mu, self.mu_target)
            self.soft_update(self.q, self.q_target)
        
        self.scheduler_mu.step()
        self.scheduler_q.step()
        self.scheduler_state.step()

    def getMemorySize(self):
        return self.memory.size() 
    
    def getParams(self):
        print("q")
        for param in self.q.parameters():
            print(param)
            print("----------------")
            print(param.grad)
            print("=================")

        print("mu")
        for param in self.mu.parameters():
            print(param)
            print("----------------")
            print(param.grad)
            print("=================")

        print("state")
        for param in self.state_representer.parameters():
            print(param)
            print("----------------")
            print(param.grad)
            print("=================")
        print("\n")
        print("=================")
        print("=================")
        print("=================")
        # self.q, self.q_target = QNet(), QNet()
        # self.mu, self.mu_target = MuNet(), MuNet()

    def getQLoss(self):
        return self.q_loss
    def getMuLoss(self):
        return self.mu_loss
