import torch.nn as nn
import torch.nn.functional as F
import math

class FCNet(nn.Module):
    def __init__(self, input_size, output_size):
        super(FCNet, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.fc1 = nn.Linear(self.input_size, 120)
        # self.fc1_drop = nn.Dropout(0.2)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, self.output_size)

    def forward(self, x):
        # reshape input data (e.g. input_size = 28*28 for mnist dataset)
        x = x.view(-1, self.input_size)
        # three fully connected layers with relu activation
        x = F.relu(self.fc1(x))
        # x = self.fc1_drop(x)
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.log_softmax(x, dim=1)
        return x

class ConvNet(nn.Module):
    def __init__(self, input_size, output_size, channel_size):
        super(ConvNet, self).__init__()
        self.input_size = input_size
        self.sqrt_input_size = int(math.sqrt(self.input_size))
        self.hidden_size = int(pow(self.sqrt_input_size / 4 - 3, 2) * 20)
        self.output_size = output_size
        self.channel_size = channel_size
        self.conv1 = nn.Conv2d(self.channel_size, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        # self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(self.hidden_size, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, self.output_size)

    def forward(self, x):
        # two convolution, max_pool layers with relu activation
        x = x.view(-1, self.channel_size, self.sqrt_input_size, self.sqrt_input_size)
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        # x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, self.hidden_size)
        # three fully connected layers with relu activation
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        # x = F.dropout(x, training=self.training)
        x = F.log_softmax(x, dim=1)
        return x

