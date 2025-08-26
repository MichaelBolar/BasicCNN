import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 5)
        self.conv2 = nn.Conv2d(32, 64, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 5 * 5, 1024)
        self.fc2 = nn.Linear(1024, 800)
        self.fc3 = nn.Linear(800, 600)
        self.fc4 = nn.Linear(600, 400)
        self.fc5 = nn.Linear(400, 200)
        self.fc6 = nn.Linear(200, 100)

    def forward(self, x):
        #print(f'This is the input {x}')
        x = self.pool(F.relu(self.conv1(x)))
        #print('pool1', x)
        #print(f'Shape of x after first conv: {x.shape}')
        x = self.pool(F.relu(self.conv2(x)))
        #print('pool2', x)
        #print(f'Shape of x after second conv: {x.shape}')
        x = torch.flatten(x, 1) #Flatten all dimensions except batch
        #print(f'Shape of x after flattening: {x.shape}')
        #print('flatten', x)
        x = F.relu(self.fc1(x))
        #print('fc1', x)
        #print(f'Shape of x after fully connected layer 1: {x.shape}')
        x = F.relu(self.fc2(x))
        #print('fc2', x)
        #print(f'Shape of x after fully connected layer 2: {x.shape}')
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        x = self.fc6(x)
        #print(f'Shape of x after Gaussian layer: {x.shape}')
        return x
    