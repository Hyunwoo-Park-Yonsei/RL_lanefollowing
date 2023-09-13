import gym
import random
import collections
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from stateRepresenter import StateRepresenter
import time

#Hyperparameters
lr_mu           = 0.05
lr_q            = 0.0001
lr_s            = 0.0005
gamma           = 0.90
batch_size      = 20
buffer_limit    = 2000
tau             = 0.001 # for target network soft update
number_of_train = 250

class ReplayBuffer():
    def __init__(self):
        self.buffer = collections.deque(maxlen=buffer_limit)

    def put(self, transition):
        self.buffer.append(transition)
    
    def sample(self, n):
        mini_batch = random.sample(self.buffer, n)
        s_lst, a_lst, r_lst, s_prime_lst, done_mask_lst = [], [], [], [], []

        for transition in mini_batch:
            s, a, r, s_prime, done = transition
            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            done_mask = 0.0 if done else 1.0 
            done_mask_lst.append([done_mask])
        
        return torch.tensor(s_lst, dtype=torch.float), torch.tensor(a_lst, dtype=torch.float), \
                torch.tensor(r_lst, dtype=torch.float), torch.tensor(s_prime_lst, dtype=torch.float), \
                torch.tensor(done_mask_lst, dtype=torch.float)
    
    def size(self):
        return len(self.buffer)

class MuNet(nn.Module):
    def __init__(self):
        super(MuNet, self).__init__()
        self.fc1 = nn.Linear(12, 2)
        # self.fc2 = nn.Linear(128, 64)
        # self.fc_mu = nn.Linear(64, 2)
        # self.fc = nn.Linear(12, 2)
        self.clipping = torch.tensor([1.0, 0.1])

    def forward(self, x):
        mu = F.relu(self.fc1(x))
        # x = F.relu(self.fc2(x))
        # mu = self.fc_mu(x)
        # mu = self.fc(x)
        # clipping = torch.tensor([0.1, 0.003])
        # mu = mu * clipping
        mu = torch.clamp(mu, min = -0.3, max = 0.3)
        mu = mu * self.clipping
        return mu

class QNet(nn.Module):
    def __init__(self):
        super(QNet, self).__init__()
        # global state_representer
        # self.state_representer = state_representer
        # self.fc_s = nn.Linear(12, 64)
        # self.fc_a = nn.Linear(2,64)
        self.fc_q = nn.Linear(14, 1)
        # self.fc_out = nn.Linear(32,1)

    def forward(self, x, a):
        # self.state_representer = state_representer
        # print("represented_state", represented_state.size())
        # h1 = F.relu(self.fc_s(x))
        # h2 = F.relu(self.fc_a(a))
        # print("h1 size", h1.size())
        # print("h2 size", h2.size())

        h1 = x.reshape(batch_size,12)
        h2 = a.reshape(batch_size,2)
        # print("h1 size", h1.size())
        # print("h2 size", h2.size())
        cat = torch.cat([h1,h2], dim=1)
        q = F.relu(self.fc_q(cat))
        # q = self.fc_out(q)
        return q

class OrnsteinUhlenbeckNoise:
    def __init__(self, mu):
        self.theta, self.dt, self.sigma = 0.01, 0.01, 0.01
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
        self.noise_ratio = 1
        

    def train(self):
        s,a,r,s_prime,done_mask  = self.memory.sample(batch_size)

        represented_state = self.state_representer(s)
        represented_state_prime = self.state_representer(s_prime)
        target = r + gamma * self.q_target(represented_state_prime, self.mu_target(represented_state_prime)) * done_mask
        self.q_loss = F.smooth_l1_loss(self.q(represented_state,a), target.detach())
        # self.q_optimizer.zero_grad()
        # q_loss.backward()
        # self.q_optimizer.step()
        
        self.mu_loss = -self.q(represented_state, self.mu(represented_state)).mean() # That's all for the policy loss.
        # self.mu_optimizer.zero_grad()
        # mu_loss.backward()
        # self.mu_optimizer.step()
        self.q_optimizer.zero_grad()
        self.mu_optimizer.zero_grad()
        self.state_optimizer.zero_grad()
        loss = self.q_loss + self.mu_loss
        loss.backward()
        # print("\n")
        # print("q")
        # for param in self.q.parameters():
        #     print(param.grad)
        # print("mu")
        # for param in self.mu.parameters():
        #     print(param.grad)
        # print("state")
        # for param in self.state_representer.parameters():
        #     print(param.grad)

        self.q_optimizer.step()
        self.mu_optimizer.step()
        self.state_optimizer.step()
        

        
    def soft_update(self, net, net_target):
        for param_target, param in zip(net_target.parameters(), net.parameters()):
            param_target.data.copy_(param_target.data * (1.0 - tau) + param.data * tau)
    
    def getAction(self, state):
        a = self.mu(self.state_representer(torch.from_numpy(state).float())) 
        action = []
        print("action ratio", self.noise_ratio, "accel noise", self.ou_noise_accel()[0], "steer", self.ou_noise_steer()[0])
        
        
        action.append(a[0][0].item() + self.noise_ratio * self.ou_noise_accel()[0])
        action.append(a[0][1].item() + self.noise_ratio * self.ou_noise_steer()[0])
        self.noise_ratio *= (1 - 1e-3)

        return action
    
    def insertMemory(self, state, action, reward, s_prime, done):
        self.memory.put((state, action ,reward/10, s_prime, done))
    
    def isMemoryFull(self):
        return self.memory.size() >= buffer_limit
    
    def startTraining(self):
        print("Start Training !")
        for i in range(number_of_train):
            self.train()
            self.soft_update(self.mu, self.mu_target)
            self.soft_update(self.q, self.q_target)

    def getMemorySize(self):
        return self.memory.size() 
    
    def getParams(self):
        print("q")
        for param in self.q.parameters():
            print(param.grad)
        # print("q target")
        # for param in self.q_target.parameters():
        #     print(param)
        print("mu")
        for param in self.mu.parameters():
            print(param.grad)
        # print("mu target")
        # for param in self.mu_target.parameters():
        #     print(param)
        print("state")
        for param in self.state_representer.parameters():
            # print(param)    
            # print("==========================================")
            print(param.grad)
        print("\n")
        # self.q, self.q_target = QNet(), QNet()
        # self.mu, self.mu_target = MuNet(), MuNet()

    def getQLoss(self):
        return self.q_loss
    def getMuLoss(self):
        return self.mu_loss

    # def main():
        # env = gym.make('Pendulum-v1', max_episode_steps=200, autoreset=True, render_mode="rgb_array")
        # print_interval = 20
        
        # for n_epi in range(10000):
        #     # s, _ = env.reset()
        #     done = False

        #     count = 0
        #     while count < 200 and not done:
                # a = mu(torch.from_numpy(s).float()) 
                # a = a.item() + ou_noise()[0]
                # s_prime, r, done, truncated, info = env.step([a])
                # memory.put((s,a,r/100.0,s_prime,done))
                # score +=r
                # s = s_prime
                # count += 1
                    
            # if memory.size()>2000:
                # for i in range(10):
                    # train(mu, mu_target, q, q_target, memory, q_optimizer, mu_optimizer)
                    # soft_update(mu, mu_target)
                    # soft_update(q,  q_target)
            
            # if n_epi%print_interval==0 and n_epi!=0:
            #     print("# of episode :{}, avg score : {:.1f}".format(n_epi, score/print_interval))
            #     score = 0.0

        # env.close()

# if __name__ == '__main__':
#     main()