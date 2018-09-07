"""
A template that creates python scripts for parallelized training and hyperparameter optimization using the ranking
loss function
"""

import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data_utils
from sklearn.metrics import f1_score
from hyperopt import hp, pyll
import warnings
from custom_mlp import MLP
import custom_loss as cl

warnings.filterwarnings("ignore")
file = open('/home/vpapaste/MLDNN/MLDNN_CPU/branch_list.pickle', 'rb')
list_of_branches = pickle.load(file)
file.close()

np.random.seed(13)
torch.manual_seed(13)
rng = np.random.RandomState(13)

space = {'num_hid_layers': hp.choice('num_hid_layers', [2, 3, 4]),
         'dropout': hp.uniform('dropout', .25, .75),
         'hidden_number_1': hp.choice('hidden_number_1', [250, 500, 800, 1000]),
         'hidden_number_2': hp.choice('hidden_number_2', [250, 500, 800, 1000]),
         'hidden_number_3': hp.choice('hidden_number_3', [250, 500, 800, 1000]),
         'hidden_number_4': hp.choice('hidden_number_4', [250, 500, 800, 1000]),
         'batch_size': hp.choice('batch_size', [50, 100, 150, 250]),
         'l_rate': hp.uniform('l_rate', -5, -1)
         }

sigmoid = nn.Sigmoid()
list_of_branches_upd = []
parameter_list = []

for sample in range(100):
	parameter_list.append(pyll.stochastic.sample(space, rng=rng))

for counter, branch in enumerate([list_of_branches[int('branch_number')]]):
	output_size = len(branch.label_names)
	parameter_dict = parameter_list[int('parameter_number') - 1]
	f1_benchmark = -1
	input_size = 258
	x_train, y_train, train_data = branch.train_tensors()
	x_valid, y_valid, valid_data = branch.valid_tensors()
	batch_size = parameter_dict['batch_size']
	train_loader = data_utils.DataLoader(train_data, batch_size=batch_size, shuffle=True)
	valid_loader = data_utils.DataLoader(valid_data, batch_size=batch_size, shuffle=True)
	lr = 10**parameter_dict['l_rate']
	net = MLP(input_size, output_size, parameter_dict)
	criterion_2 = cl.RankLoss()
	optimizer = torch.optim.SGD(net.parameters(), lr=lr , momentum=0.9, nesterov=True)
	for epoch in range(100):
		total_loss = 0
		for i, data in enumerate(train_loader, 0):
			inputs, labels = data
			optimizer.zero_grad()
			outputs = net(inputs)
			loss = criterion_2(outputs, labels)
			loss.backward()
			optimizer.step()
			total_loss += loss.item()
		net.eval()
		valid_predictions = sigmoid(net(x_valid)).round()
		valid_f1_median = np.median(f1_score(y_valid.detach().numpy(), valid_predictions.detach().numpy(), average=None))
		net.train()
		if valid_f1_median > f1_benchmark:
			f1_benchmark = valid_f1_median
			torch.save(net.state_dict(), str(branch.go_term) + 'LTR'+ 'parameter_number' +'.pt')
			branch.parameter_dict = parameter_dict
			branch.best_tuple = (branch.go_term, sample, epoch)
			branch.best_f1 = f1_benchmark
list_of_branches_upd.append(branch)
file = open(str(branch.go_term) + 'parameter_number' + '_trained_LTR.pickle', 'wb')
pickle.dump(list_of_branches_upd, file)
file.close()