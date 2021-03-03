from torch import nn
import torch.nn.functional as F
import torch

class Net(nn.Module):
    def __init__(self, inputs, hidden_layer_size, outputs):
        super(Net,self).__init__()
        self.input_layer = nn.Linear(inputs, hidden_layer_size)
        self.hidden_layer = nn.Linear(hidden_layer_size, outputs)

    def forward(self, x):
        x = F.relu(self.input_layer(x))
        x = self.hidden_layer(x)
        return x
