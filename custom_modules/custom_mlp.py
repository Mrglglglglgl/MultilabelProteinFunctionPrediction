"""
a class representing a multilayer perceptron with custom number of hidden layers and hidden units
"""

import torch.nn as nn
import torch.nn.functional as F
import torch
import warnings
warnings.filterwarnings("ignore")
torch.manual_seed(13)

class MLP(nn.Module):
   def __init__(self, input_size, output_size, parameter_dict):
      super(MLP, self).__init__()
      self.nodes_list = [parameter_dict['hidden_number_1'], parameter_dict['hidden_number_2'], parameter_dict['hidden_number_3'], parameter_dict['hidden_number_4']]
      self.dropout =  parameter_dict['dropout']
      self.num_hid_layers =  parameter_dict['num_hid_layers']                                         
      self.net = nn.Sequential()
      self.net.add_module('input_layer', nn.Linear(input_size, self.nodes_list[0]))
      self.net.add_module('input_bn', nn.BatchNorm1d(self.nodes_list[0]))
      for layer in range(parameter_dict['num_hid_layers'] - 1):
         self.net.add_module('hidden_'+ str(layer), nn.Linear(self.nodes_list[layer], self.nodes_list[layer+1]))
         self.net.add_module('hidden_'+ str(layer)+'bn', nn.BatchNorm1d(self.nodes_list[layer+1]))
      self.net.add_module('output_layer', nn.Linear(self.nodes_list[parameter_dict['num_hid_layers'] - 1],output_size))


   def forward(self, x):
        for layer in range(0, 2*self.num_hid_layers , 2):
            x = self.net[layer](x)
            x = self.net[layer+1](x)
            x = F.relu(x)
            x = F.dropout(x, training=self.training, p=self.dropout)
        x = self.net[-1](x)
        return x