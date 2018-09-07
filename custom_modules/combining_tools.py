"""
a method containing function to be used for combining MLDNN predictions
for the different cost functions
"""

import pickle
import pandas as pd
import torch
from collections import Counter, defaultdict
import torch.nn as nn
from custom_mlp import MLP

def combine_branches_ltr_only(x_test, x_test_df, fmax_ltr_only_idx_list):

	file = open('/cluster/project1/FFPredLTR/MLDNN/branch_list.pickle', 'rb')
	list_of_branches = pickle.load(file)
	file.close()
	sigmoid = nn.Sigmoid().cuda()
	evaluation = defaultdict()
	input_size = 258
	for idx, (branch, margin) in enumerate(zip(list_of_branches, fmax_ltr_only_idx_list)):
		output_size = len(branch.label_names)
		file = open('/cluster/project1/FFPredLTR/MLDNN/LTR_only_experiments_margin_' + margin + '/branches_metrics_LTR_only_margin_' + margin + '_light.pickle', 'rb')
		trained_branch = pickle.load(file)[idx]
		net = MLP(input_size, output_size, trained_branch.parameter_dict).cuda()
		net.load_state_dict(torch.load('/cluster/project1/FFPredLTR/MLDNN/LTR_only_experiments_margin_' + margin + '/parameters/' + branch.go_term + 'LTR_only_margin_'+ margin + '.pt'))
		net.eval()
		test_probabilities = sigmoid(net(x_test)).detach().cpu().numpy()
		df = pd.DataFrame(index=x_test_df.index, columns=trained_branch.label_names, data=test_probabilities)
		evaluation[trained_branch.go_term] = df
	file = open('final_probabilities_per_branch_ltr_only.pickle', 'wb')
	pickle.dump(evaluation, file)
	file.close()

def final_predictions_ltr_only(x_test, x_test_df):

	metrics_dict = defaultdict()
	file = open('bp_terms_dict.pickle', 'rb')
	bp_terms_dict = pickle.load(file)
	file.close()
	file = open('mf_terms_dict.pickle', 'rb')
	mf_terms_dict = pickle.load(file)
	file.close()
	file = open('cc_terms_dict.pickle', 'rb')
	cc_terms_dict = pickle.load(file)
	file.close()
	file = open('final_probabilities_per_branch_ltr_only.pickle', 'rb')
	combined_dict = pickle.load(file)
	file.close()
	list_of_go_terms = list(bp_terms_dict.keys()) + list(mf_terms_dict.keys()) + list(cc_terms_dict.keys())
	final_probabilities_df = pd.DataFrame(0, index= x_test_df.index, columns=list_of_go_terms)
	for protein in list(final_probabilities_df.index):
		for branch_name, branch_preds_dataframe in combined_dict.items():
			for label in list(branch_preds_dataframe.columns):
				final_probabilities_df.loc[protein, label] += branch_preds_dataframe.loc[protein, label]
	for name, freq in bp_terms_dict.items():
		final_probabilities_df[str(name)] = final_probabilities_df[str(name)].div(freq).round(2)
	for name, freq in mf_terms_dict.items():
		final_probabilities_df[str(name)] = final_probabilities_df[str(name)].div(freq).round(2)
	for name, freq in cc_terms_dict.items():
		final_probabilities_df[str(name)] = final_probabilities_df[str(name)].div(freq).round(2)
	final_predictions_df = final_probabilities_df.round(0)

	metrics_dict['preds'] = final_predictions_df
	metrics_dict['probs'] = final_probabilities_df

	file = open('final_evaluation_dict_ltr_only.pickle', 'wb')
	pickle.dump(metrics_dict, file)
	file.close()

def combine_branches_ltr(x_test, x_test_df, fmax_ltr_idx_list):

	file = open('/cluster/project1/FFPredLTR/MLDNN/branch_list.pickle', 'rb')
	list_of_branches = pickle.load(file)
	file.close()
	sigmoid = nn.Sigmoid().cuda()
	evaluation = defaultdict()
	input_size = 258
	for idx, (branch, margin) in enumerate(zip(list_of_branches, fmax_ltr_idx_list)):
		output_size = len(branch.label_names)
		file = open('/cluster/project1/FFPredLTR/MLDNN/LTR_experiments_margin_' + '10' + '/branches_metrics_LTR_margin_' + '10' + '_light.pickle', 'rb')
		trained_branch = pickle.load(file)[idx]
		net = MLP(input_size, output_size, trained_branch.parameter_dict).cuda()
		net.load_state_dict(torch.load('/cluster/project1/FFPredLTR/MLDNN/LTR_experiments_margin_' + '10' + '/parameters/' + branch.go_term + 'LTR_margin_'+ '10' + '.pt'))
		net.eval()
		test_probabilities = sigmoid(net(x_test)).detach().cpu().numpy()
		df = pd.DataFrame(index=x_test_df.index, columns=trained_branch.label_names, data=test_probabilities)
		evaluation[trained_branch.go_term] = df
	file = open('final_probabilities_per_branch_ltr.pickle', 'wb')
	pickle.dump(evaluation, file)
	file.close()

def final_predictions_ltr(x_test, x_test_df):

	metrics_dict = defaultdict()
	file = open('bp_terms_dict.pickle', 'rb')
	bp_terms_dict = pickle.load(file)
	file.close()
	file = open('mf_terms_dict.pickle', 'rb')
	mf_terms_dict = pickle.load(file)
	file.close()
	file = open('cc_terms_dict.pickle', 'rb')
	cc_terms_dict = pickle.load(file)
	file.close()
	file = open('final_probabilities_per_branch_ltr.pickle', 'rb')
	combined_dict = pickle.load(file)
	file.close()
	list_of_go_terms = list(bp_terms_dict.keys()) + list(mf_terms_dict.keys()) + list(cc_terms_dict.keys())
	final_probabilities_df = pd.DataFrame(0, index=x_test_df.index, columns=list_of_go_terms)
	print(x_test_df.index)
	for protein_idx, protein in enumerate(list(final_probabilities_df.index)):
		for branch_name, branch_preds_dataframe in combined_dict.items():
			for label_idx, label in enumerate(list(branch_preds_dataframe.columns)):
				increment = branch_preds_dataframe.loc[protein,label]
				final_probabilities_df.loc[protein,label] += increment
	for name, freq in bp_terms_dict.items():
		final_probabilities_df[str(name)] = final_probabilities_df[str(name)].div(freq).round(2)
	for name, freq in mf_terms_dict.items():
		final_probabilities_df[str(name)] = final_probabilities_df[str(name)].div(freq).round(2)
	for name, freq in cc_terms_dict.items():
		final_probabilities_df[str(name)] = final_probabilities_df[str(name)].div(freq).round(2)
	final_predictions_df = final_probabilities_df.round(0)
	metrics_dict['preds'] = final_predictions_df
	metrics_dict['probs'] = final_probabilities_df
	file = open('final_evaluation_dict_ltr.pickle', 'wb')
	pickle.dump(metrics_dict, file)
	file.close()


def combine_branches_bce(x_test, x_test_df):

	file = open('/cluster/project1/FFPredLTR/MLDNN/branch_list.pickle', 'rb')
	list_of_branches = pickle.load(file)
	file.close()
	sigmoid = nn.Sigmoid().cuda()
	evaluation = defaultdict()
	input_size = 258
	for idx, branch in enumerate(list_of_branches):
		output_size = len(branch.label_names)
		file = open('/cluster/project1/FFPredLTR/MLDNN/BCE_experiments/branches_metrics_bce_light.pickle', 'rb')
		trained_branch = pickle.load(file)[idx]
		net = MLP(input_size, output_size, trained_branch.parameter_dict).cuda()
		net.load_state_dict(torch.load('/cluster/project1/FFPredLTR/MLDNN/BCE_experiments/parameters/'+ branch.go_term + '.pt'))
		net.eval()
		test_probabilities = sigmoid(net(x_test)).detach().cpu().numpy()
		df = pd.DataFrame(index=x_test_df.index, columns=trained_branch.label_names, data=test_probabilities)
		evaluation[trained_branch.go_term] = df
	file = open('final_probabilities_per_branch_bce.pickle', 'wb')
	pickle.dump(evaluation, file)
	file.close()

def final_predictions_bce(x_test, x_test_df):

	metrics_dict = defaultdict()
	file = open('bp_terms_dict.pickle', 'rb')
	bp_terms_dict = pickle.load(file)
	file.close()
	file = open('mf_terms_dict.pickle', 'rb')
	mf_terms_dict = pickle.load(file)
	file.close()
	file = open('cc_terms_dict.pickle', 'rb')
	cc_terms_dict = pickle.load(file)
	file.close()
	file = open('final_probabilities_per_branch_bce.pickle', 'rb')
	combined_dict = pickle.load(file)
	file.close()
	list_of_go_terms = list(bp_terms_dict.keys()) + list(mf_terms_dict.keys()) + list(cc_terms_dict.keys())
	final_probabilities_df = pd.DataFrame(0, index= x_test_df.index, columns=list_of_go_terms)
	for protein in list(final_probabilities_df.index):
		for branch_name, branch_preds_dataframe in combined_dict.items():
			for label in list(branch_preds_dataframe.columns):
				final_probabilities_df.loc[protein, label] += branch_preds_dataframe.loc[protein, label]
	for name, freq in bp_terms_dict.items():
		final_probabilities_df[str(name)] = final_probabilities_df[str(name)].div(freq).round(2)
	for name, freq in mf_terms_dict.items():
		final_probabilities_df[str(name)] = final_probabilities_df[str(name)].div(freq).round(2)
	for name, freq in cc_terms_dict.items():
		final_probabilities_df[str(name)] = final_probabilities_df[str(name)].div(freq).round(2)
	final_predictions_df = final_probabilities_df.round(0)
	metrics_dict['preds'] = final_predictions_df
	metrics_dict['probs'] = final_probabilities_df
	file = open('final_evaluation_dict_bce.pickle', 'wb')
	pickle.dump(metrics_dict, file)
	file.close()