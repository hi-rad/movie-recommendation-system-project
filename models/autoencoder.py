import torch.nn as nn


class AutoEncoder(nn.Module):
    def __init__(self, num_features, dimensions=None):
        super(AutoEncoder, self).__init__()

        if dimensions is None:
            dimensions = [128, 64, 128]

        self.layers = nn.ModuleList()
        input_dim = num_features
        for layer_num in range(len(dimensions) + 1):
            # Last Layer
            if layer_num == len(dimensions):
                self.layers.append(nn.Linear(input_dim, num_features))
            else:
                self.layers.append(nn.Linear(input_dim, dimensions[layer_num]))
                input_dim = dimensions[layer_num]
        self.activation = nn.ReLU()

    def forward(self, x):
        for layer_num in range(len(self.layers)):
            # No Activation for the Last Layer
            if layer_num == len(self.layers) - 1:
                x = self.layers[layer_num](x)
            else:
                x = self.activation(self.layers[layer_num](x))
        return x
