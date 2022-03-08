import numpy as np
import torch
import torch.nn as nn
import torch.optim as opt
import torch.nn.functional as F

class CA_Model(nn.Module):
    def __init__(self, channel_n, fire_rate, device, hidden_size=128):
        super(CA_Model, self).__init__()
        self.device = device
        self.channel_n = channel_n

        self.fc0 = nn.Linear(channel_n*3, hidden_size)
        self.fc1 = nn.Linear(hidden_size, channel_n, bias=False)
        with torch.no_grad():
            self.fc1.weight.zero_()

        self.fire_rate = fire_rate
        self.to(self.device)

    def alive(self, x):  # a cell is alive if the max neighbor is greater than 0.1
        return F.max_pool2d(x[:, 3:4, :, :], kernel_size=3, stride=1, padding=1) > 0.1

    def perceive(self, x, angle):

        def _perceive_with(x, filter):
            conv_filter = torch.from_numpy(filter.astype(np.float32)).to(self.device)
            conv_filter = conv_filter.view(1,1,3,3).repeat(self.channel_n,1,1,1)
            # since these is a one-channel filter (duplicated), it only applies to a single input channel, do not combine them
            return F.conv2d(x, conv_filter, padding=1, groups=self.channel_n)

        dx = np.outer([1,2,1], [-1,0,1])/8.0  # sobel filter
        dy = dx.T
        c = np.cos(angle*np.pi/180)
        s = np.sin(angle*np.pi/180)
        f1 = c*dx - s*dy
        f2 = s*dx + c*dy

        y1= _perceive_with(x,f1)
        y2= _perceive_with(x,f2)
        y = torch.cat((x,y1,y2),1)
        return y

    def update(self, x, fire_rate, angle):
        x = x.transpose(1,3)
        pre_life_mask = self.alive(x)
        dx = self.perceive(x, angle)
        dx = dx.transpose(1,3)
        dx = self.fc0(dx)
        dx = F.relu(dx)
        dx = self.fc1(dx)

        if fire_rate is None:
            fire_rate=self.fire_rate
        # stochastic = torch.rand([dx.size(0), dx.size(1), dx.size(2), 1])>fire_rate
        # stochastic = stochastic.float().to(self.device)
        # dx = dx * stochastic

        x = x+dx.transpose(1,3)

        post_life_mask = self.alive(x)
        life_mask = (pre_life_mask & post_life_mask).float()
        x = x*life_mask
        return x.transpose(1,3)

    def forward(self, x, steps=1, fire_rate=None, angle=0.0):
        for step in range(steps):
            x = self.update(x, fire_rate, angle)
        return x


