import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class StateRepresenter(nn.Module):
  def __init__(self):
    super(StateRepresenter, self).__init__()
    self.conv1 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=(1, 1))
    nn.init.kaiming_normal_(self.conv1.weight, nonlinearity='relu')
    self.conv2 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=(1, 1))
    nn.init.kaiming_normal_(self.conv2.weight, nonlinearity='relu')
    # self.conv3 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, stride=2, padding=(1, 1))
    # self.conv4 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, stride=2, padding=(1, 1))
    self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    self.fc1 = nn.Linear(512, 128)
    nn.init.kaiming_normal_(self.fc1.weight, nonlinearity='relu')
    self.fc2 = nn.Linear(128,12)
    nn.init.kaiming_normal_(self.fc2.weight, nonlinearity='relu')
    self.ELU = torch.nn.ELU()
  
  def forward(self, x):
    x = F.normalize(x, dim=0)
    # print("연산 전", x.size())
    x = self.pool(F.relu(self.conv1(x)))
    # print("conv1 연산 후", x.size())
    x = self.pool(F.relu(self.conv2(x)))
    # print("conv2 연산 후",x.size())
    # x = self.pool(F.relu(self.conv3(x)))
    # print("conv3 연산 후",x.size())
    # x = self.pool(F.relu(self.conv4(x)))  
    # print("conv4 연산 후",x.size())
    x = x.view(x.size(0), -1) # flatten
    # print("x", x.size())
    x = F.relu(self.fc1(x))
    x = F.relu(self.fc2(x))
    # print("final output", x.size())
    return x

# cnn_test = StateRepresenter()
# cnn_test.forward(torch.randn(1, 1, 256, 128))