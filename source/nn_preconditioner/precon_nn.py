# here we will train a neural network to predict k_alpha and k_psi, given w_n, w_T, w_alpha, and w_psi
# The network is a simple feedforward neural network with N hidden layers
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# Set the device to MPS
mps_device = torch.device("mps")

# Define the neural network
class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_size, number_layers, output_size):
        super(SimpleNN, self).__init__()
        self.hidden_layers = nn.ModuleList()
        self.hidden_layers.append(nn.Linear(input_size, hidden_size))
        for _ in range(number_layers-1):  # number_layers hidden layers
            self.hidden_layers.append(nn.Linear(hidden_size, hidden_size))
        self.output_layer = nn.Linear(hidden_size, output_size)
        self.activation = nn.Softplus()

    def forward(self, x):
        # rescale the input by its norm
        norm = torch.norm(x, dim=1).reshape(-1, 1)
        x = x / norm
        for layer in self.hidden_layers:
            x = self.activation(layer(x))
        x = self.output_layer(x)
        return x * norm
