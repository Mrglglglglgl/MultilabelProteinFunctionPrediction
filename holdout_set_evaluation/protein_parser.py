
# now we have the protein array and the list that holds which branch had the best fmax performance
# we need to get the corresponding networks for these margin values and put them into a list

import sys
sys.path.insert(0, '/cluster/project1/FFPredLTR/MLDNN/custom_modules')
import os
import pickle
import pandas as pd
import numpy as np
import torch
from collections import Counter, defaultdict
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, matthews_corrcoef
import warnings
from custom_mlp import MLP
import pickle

warnings.filterwarnings("ignore")
file = open('bp_terms_dict.pickle', 'rb')
bp_terms_dict = pickle.load(file)
file.close()
file = open('mf_terms_dict.pickle', 'rb')
mf_terms_dict = pickle.load(file)
file.close()
file = open('cc_terms_dict.pickle', 'rb')
cc_terms_dict = pickle.load(file)
file.close()

data_df = pd.read_csv('/cluster/project1/FFPredRNN/MLDNN/reference_goa_nk.txt', sep='\t', names=['Proteins', 'Labels'])
all_proteins = data_df["Proteins"]
labels = data_df["Labels"]
proteins = all_proteins.unique()

set_labels = list(labels.unique())
branch_labels = list(bp_terms_dict.keys()) + list(mf_terms_dict.keys()) + list(cc_terms_dict.keys())

file = open('/cluster/project1/FFPredLTR/MLDNN/branch_list.pickle', 'rb')
list_of_branches = pickle.load(file)
file.close()


feature_list = []
for protein in proteins:
	feature_list_per_protein = []
	lines = open('/cluster/project1/FFPredLTR/CAFA3Targets/' + protein + '.features').readlines()
	lines = lines[0].split('\t')
	for line in lines[1:-1]:
		position = line.find(':')
		feature_list_per_protein.append(float(line[position + 1:]))
	feature_list.append(feature_list_per_protein)

protein_df = pd.DataFrame(data=feature_list, index=proteins)
protein_array = np.array(feature_list)
protein_tensor = torch.cuda.FloatTensor(protein_array)


per_loss_fmax_dict = defaultdict()

for margin in ['00', '05', '10']:
	fmax_bce_list = []
	fmax_ltr_list = []
	fmax_ltr_only_list = []

	for index, branch in enumerate(list_of_branches):
		df_bce = pd.read_csv('/cluster/project1/FFPredLTR/Output_Fmax/BCLoss/' + str(branch.go_term) + '_preds.csv.pr_results', skiprows=4)
		if margin == '00':
			df_ltr = pd.read_csv('/cluster/project1/FFPredLTR/Output_Fmax/LTRLoss/' + margin + '/' + str(branch.go_term) + '_preds_LTR.csv.pr_results', skiprows=4)
		else:
			df_ltr = pd.read_csv('/cluster/project1/FFPredLTR/Output_Fmax/LTRLoss/' + margin + '/' + str(branch.go_term) + '_preds_LTR_margin_' + margin + '.csv.pr_results', skiprows=4)
		df_ltr_only = pd.read_csv('/cluster/project1/FFPredLTR/Output_Fmax/LTROnlyLoss/' + margin + '/' + str(branch.go_term) + '_preds_LTR_only_margin_' + margin + '.csv.pr_results', skiprows=4)

		# get bce fmax
		fmax_bce = df_bce['F1']
		best_fmax_bce = np.max(fmax_bce)
		fmax_bce_list.append(best_fmax_bce)

		# get ltr fmax
		fmax_ltr = df_ltr['F1']
		best_fmax_ltr = np.max(fmax_ltr)
		fmax_ltr_list.append(best_fmax_ltr)	
		
		# get ltronly fmax
		fmax_ltr_only = df_ltr_only['F1']
		best_fmax_ltr_only = np.max(fmax_ltr_only)
		fmax_ltr_only_list.append(best_fmax_ltr_only)
		
	per_loss_fmax_dict['bce'] = fmax_bce_list
	per_loss_fmax_dict['ltr_' + margin] = fmax_ltr_list
	per_loss_fmax_dict['ltr_only_' + margin] = fmax_ltr_only_list

fmax_ltr_list = list(np.array([per_loss_fmax_dict['ltr_00'], per_loss_fmax_dict['ltr_05'], per_loss_fmax_dict['ltr_10']]).max(axis=0))

fmax_ltr_idx_list = list(np.array([per_loss_fmax_dict['ltr_00'], per_loss_fmax_dict['ltr_05'], per_loss_fmax_dict['ltr_10']]).argmax(axis=0))

fmax_ltr_only_list = list(np.array([per_loss_fmax_dict['ltr_only_00'], per_loss_fmax_dict['ltr_only_05'], per_loss_fmax_dict['ltr_only_10']]).max(axis=0))

fmax_ltr_only_idx_list = list(np.array([per_loss_fmax_dict['ltr_only_00'], per_loss_fmax_dict['ltr_only_05'], per_loss_fmax_dict['ltr_only_10']]).argmax(axis=0))

fmax_ltr_idx_list = ['00' if x==0 else '05' if x==1 else '10' for x in fmax_ltr_idx_list]

fmax_ltr_only_idx_list = ['00' if x==0 else '05' if x==1 else '10' for x in fmax_ltr_only_idx_list]

file = open('fmax_best_margins_list_ltr.pickle', 'wb')
pickle.dump(fmax_ltr_idx_list, file)
file.close()

file = open('fmax_best_margins_list_ltr_only.pickle', 'wb')
pickle.dump(fmax_ltr_only_idx_list, file)
file.close()

file = open('protein_cuda_tensor.pickle', 'wb')
pickle.dump(protein_tensor, file)
file.close()

file = open('protein_df.pickle', 'wb')
pickle.dump(protein_df, file)
file.close()