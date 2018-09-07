"""
a class representing a branch (MLDNN), contaning information regarding the branch, its datasets and metrics and some useful methods. 
Ideally it needs to be split into smaller classes.
"""

import pandas as pd
import numpy as np
import torch
from torchsample import TensorDataset
import warnings

warnings.filterwarnings("ignore")

torch.manual_seed(13)

class Branch(object):
    
    def __init__(self, go_term, dict_of_go_terms, dict_of_categories):
        self.go_term = go_term
        self.category = dict_of_categories[go_term]
        self.x_train = dict_of_go_terms[go_term]['x_train']
        self.y_train = dict_of_go_terms[go_term]['y_train']
        self.x_valid = dict_of_go_terms[go_term]['x_valid']
        self.y_valid = dict_of_go_terms[go_term]['y_valid']
        self.x_test = dict_of_go_terms[go_term]['x_test']
        self.y_test = dict_of_go_terms[go_term]['y_test']
        self.label_names = dict_of_go_terms[go_term]['y_test'].columns.values.tolist()[1:]
        self.f1_median_valid = -1.0
        self.f1_median_test = -1.0
        
    def train_size(self):

        return self.x_train.shape

    def valid_size(self):

        return self.x_valid.shape

    def test_size(self):

        return self.x_test.shape

    def train_tensors(self):
        x_train_tensor = torch.FloatTensor((self.x_train.values[:, 1:]).astype(np.float32))
        (self.y_train).replace(np.inf, 0, inplace=True)
        y_train_tensor = torch.FloatTensor((self.y_train.values[:, 1:]).astype(np.int))
        
        return x_train_tensor, y_train_tensor, TensorDataset(x_train_tensor, y_train_tensor)
    
    def valid_tensors(self):
        x_valid_tensor = torch.FloatTensor((self.x_valid.values[:, 1:]).astype(np.float32))
        (self.y_valid).replace(np.inf, 0, inplace=True)
        y_valid_tensor = torch.FloatTensor((self.y_valid.values[:, 1:]).astype(np.int))

        return x_valid_tensor, y_valid_tensor, TensorDataset(x_valid_tensor, y_valid_tensor)
         
    def test_tensors(self):
        x_test_tensor = torch.FloatTensor((self.x_test.values[:, 1:]).astype(np.float32))
        (self.y_test).replace(np.inf, 0, inplace=True)
        y_test_tensor = torch.FloatTensor((self.y_test.values[:, 1:]).astype(np.int))
        
        return x_test_tensor, y_test_tensor, TensorDataset(x_test_tensor, y_test_tensor)


    def get_prediction_probabilities(self, predictions, t=1):

        if t==0:
            prediction_probabilities = pd.DataFrame(predictions, index=self.y_train["Unnamed: 0"], columns=self.label_names)
        elif t==1:
            prediction_probabilities = pd.DataFrame(predictions, index=self.y_test["Unnamed: 0"],columns=self.label_names)
        else:
            prediction_probabilities = pd.DataFrame(predictions, index=self.y_valid["Unnamed: 0"], columns=self.label_names)

        return prediction_probabilities
