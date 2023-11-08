import torch
from torch import nn

class LeNet(nn.Module):
    def __init__(self):
        super().__init__()
        ###################################################################
        # TODO: Design your own network, define layers here.              #
        # Some common Choices are: Linear, Conv2d, ReLU, MaxPool2d        #
        ###################################################################
        # Replace "pass" statement with your code
        self.conv1 = nn.Conv2d(1, 20, 5,padding = 2)
        self.conv2 = nn.Conv2d(20, 50, 5, padding = 2)
        #full connected layer
        self.fc1 = nn.Linear(50 * 7 * 7, 500) 
        self.fc2 = nn.Linear(500, 10)
        #reLu function
        self.relu = nn.ReLU()

        #maxpool2d 
        self.pool = nn.MaxPool2d(kernel_size = 2, stride= 2)
        ###################################################################
        #                     END OF YOUR CODE                            #
        ###################################################################

    def forward(self,x):
        ###################################################################
        # TODO: Design your own network, implement forward pass here.     #
        ###################################################################
        # Replace "pass" statement with your code
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        
        ###################################################################
        #                     END OF YOUR CODE                            #
        ###################################################################
        return x