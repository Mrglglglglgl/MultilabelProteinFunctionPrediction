
"""
a script that loads the best model developed using the 
BCE loss to get the the metrics on the test set
"""

import sys
sys.path.insert(0, '/cluster/project1/FFPredLTR/MLDNN/custom_modules')
import pickle
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, matthews_corrcoef
from hyperopt import hp, pyll
from custom_mlp import MLP
import warnings

warnings.filterwarnings("ignore")

file = open('branch_list.pickle', 'rb')
list_of_branches = pickle.load(file)
file.close()
np.random.seed(13)
torch.manual_seed(13)
rng = np.random.RandomState(13)
sigmoid = nn.Sigmoid()
trained_branches_metrics = []
precisions = []
recalls = []
f1_scores = []

for counter, branch in enumerate(list_of_branches):
	mccs = []
	file = open('/cluster/project1/FFPredRNN/MLDNN/BCE/'+ str(branch.go_term) + 'trained_BCE.pickle', 'rb')
	trained_branch = pickle.load(file)
	file.close()
	x_test, y_test, _ = trained_branch.test_tensors()
	input_size = 258
	output_size = len(trained_branch.label_names)
	net = MLP(input_size, output_size, trained_branch.parameter_dict).cuda()
	net.load_state_dict(torch.load('/cluster/project1/FFPredRNN/MLDNN/BCE/' + str(branch.go_term) + 'BCE.pt'))
	net.eval()
	trained_branch.test_probabilities = sigmoid(net(x_test)).detach().cpu().numpy()
	trained_branch.test_predictions = sigmoid(net(x_test)).round().detach().cpu().numpy()
	trained_branch.f1_scores = f1_score(y_test.detach().cpu().numpy(), trained_branch.test_predictions, average=None)
	trained_branch.f1_median = np.median(trained_branch.f1_scores)
	trained_branch.recall_scores = recall_score(y_test.detach().cpu().numpy(), trained_branch.test_predictions, average=None)
	trained_branch.recall_median = np.median(trained_branch.recall_scores)
	trained_branch.precision_scores = precision_score(y_test.detach().cpu().numpy(), trained_branch.test_predictions, average=None)
	trained_branch.precision_median = np.median(trained_branch.precision_scores)
	for goterm in range(output_size):
		mccs.append(matthews_corrcoef(list(y_test[:,goterm]), list(trained_branch.test_predictions[:,goterm])))
	trained_branch.mcc_scores = mccs
	trained_branch.median_mcc = np.median(mccs)
	trained_branch.roc_auc = roc_auc_score(y_test.detach().cpu().numpy(), trained_branch.test_probabilities, average=None)
	trained_branch.roc_auc_median = np.median(trained_branch.roc_auc)
	trained_branches_metrics.append(trained_branch)
file = open('/cluster/project1/FFPredRNN/MLDNN/BCE/branches_metrics_BCE.pickle', 'wb')
pickle.dump(trained_branches_metrics, file)
file.close()
print("Test Metrics Done!")