import torch
import torch.nn as nn

class DenseModel(nn.Module):
        def __init__(self, input_size: int ,hidden_layers: int,layer_width: int,  output_size: int):
            super(DenseModel,self).__init__()
            self.input_size = input_size
            self.first_layer = nn.Flatten()
            self.hidden_layers = hidden_layers
            self.layer_width = layer_width
            self.output_size = output_size


            # Layers
            self.layers = nn.ModuleList()
            last_size = input_size
            for i in range(self.hidden_layers):
                self.layers.append(nn.Linear(last_size, self.layer_width))
                self.layers.append(nn.ReLU())
                last_size = self.layer_width
                if i % 5 == 0:
                    self.layers.append(nn.Dropout(0.2))


            # Output layer
            self.layers.append(nn.Linear(last_size, self.output_size))



        def forward(self, x):
            x = self.first_layer(x)
            for layer in self.layers:
                x = layer(x)
            return x
        