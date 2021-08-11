import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

class GenericNetwork(nn.Module):
    def __init__(self, input_dimensions, layer1_dimensions, layer2_dimensions, output_dimensions, ALPHA):
        super(GenericNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dimensions, layer1_dimensions)
        self.fc2 = nn.Linear(layer1_dimensions, layer2_dimensions)
        self.output_layer = nn.Linear(layer2_dimensions, output_dimensions)

        self.optimizer = optim.Adam(self.parameters(), lr = ALPHA)
        self.device = "cuda" if T.cuda.is_available() else "cpu"

    def forward(self, input):
        input = T.FloatTensor(input).to(self.device)
        y = F.relu(self.fc1(input))
        y = F.relu(self.fc2(y))
        output = self.output_layer(y)
        return output
