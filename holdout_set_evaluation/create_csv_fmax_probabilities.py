
"""
a script that creates csv files to be used for the calculation of the holdout set Fmax probabilities
"""

import pickle
import pandas as pd
import numpy as np


loss = 'all'

if loss == 'bce':

	file = open('final_evaluation_dict_bce.pickle', 'rb')
	final_eval_dict = pickle.load(file)
	file.close()

	probs_df = final_eval_dict['probs']
	protein_names_list = []
	go_term_names_list = []
	probs_list = []
	for protein_idx, protein in enumerate(list(probs_df.index)):
		for go_term_idx, go_term in enumerate(list(probs_df.columns)):
			if probs_df.iloc[protein_idx, go_term_idx] >= 0.01:
				protein_names_list.append(protein)
				mod_term = list(go_term)
				mod_term.insert(2, ':')
				mod_term = ''.join(mod_term)
				go_term_names_list.append(mod_term)
				probs_list.append(probs_df.iloc[protein_idx, go_term_idx])
	final_eval_df = pd.DataFrame(data={'Proteins': protein_names_list, 'GO_terms': go_term_names_list, 'Probabilities': probs_list})
	final_eval_df.to_csv('holdout_preds_bce.csv', sep=' ', float_format='%.2f', header=False, index=False)

	labels = pd.read_csv('/cluster/project1/FFPredLTR/MLDNN/final_evaluation/reference_goa_nk.txt', sep='\t', header=None)
	labels.to_csv('holdout_labels.csv', sep=' ', float_format='%.2f', header=False, index=False)

elif loss == 'ltr'

	file = open('final_evaluation_dict_ltr.pickle', 'rb')
	final_eval_dict = pickle.load(file)
	file.close()

	probs_df = final_eval_dict['probs']
	protein_names_list = []
	go_term_names_list = []
	probs_list = []

	for protein_idx, protein in enumerate(list(probs_df.index)):
		for go_term_idx, go_term in enumerate(list(probs_df.columns)):
			if probs_df.iloc[protein_idx, go_term_idx] >= 0.01:
				protein_names_list.append(protein)
				mod_term = list(go_term)
				mod_term.insert(2, ':')
				mod_term = ''.join(mod_term)
				go_term_names_list.append(mod_term)
				probs_list.append(probs_df.iloc[protein_idx, go_term_idx])
	final_eval_df = pd.DataFrame(data={'Proteins': protein_names_list, 'GO_terms': go_term_names_list, 'Probabilities': probs_list})
	final_eval_df.to_csv('holdout_preds_ltr.csv', sep=' ', float_format='%.2f', header=False, index=False)

	labels = pd.read_csv('/cluster/project1/FFPredLTR/MLDNN/final_evaluation/reference_goa_nk.txt', sep='\t', header=None)
	labels.to_csv('holdout_labels.csv', sep=' ', float_format='%.2f', header=False, index=False)


elif loss == 'ltr_only':

	file = open('final_evaluation_dict_ltr_only.pickle', 'rb')
	final_eval_dict = pickle.load(file)
	file.close()

	probs_df = final_eval_dict['probs']
	protein_names_list = []
	go_term_names_list = []
	probs_list = []

	for protein_idx, protein in enumerate(list(probs_df.index)):
		for go_term_idx, go_term in enumerate(list(probs_df.columns)):
			if probs_df.iloc[protein_idx, go_term_idx] >= 0.01:
				protein_names_list.append(protein)
				mod_term = list(go_term)
				mod_term.insert(2, ':')
				mod_term = ''.join(mod_term)
				go_term_names_list.append(mod_term)
				probs_list.append(probs_df.iloc[protein_idx, go_term_idx])
	final_eval_df = pd.DataFrame(data={'Proteins': protein_names_list, 'GO_terms': go_term_names_list, 'Probabilities': probs_list})
	final_eval_df.to_csv('holdout_preds_ltr_only.csv', sep=' ', float_format='%.2f', header=False, index=False)


else:

	file = open('final_evaluation_dict_bce.pickle', 'rb')
	final_eval_dict = pickle.load(file)
	file.close()

	probs_df = final_eval_dict['probs']
	protein_names_list = []
	go_term_names_list = []
	probs_list = []

	for protein_idx, protein in enumerate(list(probs_df.index)):
		for go_term_idx, go_term in enumerate(list(probs_df.columns)):
			if probs_df.iloc[protein_idx, go_term_idx] >= 0.01:
				protein_names_list.append(protein)
				mod_term = list(go_term)
				mod_term.insert(2, ':')
				mod_term = ''.join(mod_term)
				go_term_names_list.append(mod_term)
				probs_list.append(probs_df.iloc[protein_idx, go_term_idx])
	final_eval_df = pd.DataFrame(data={'Proteins': protein_names_list, 'GO_terms': go_term_names_list, 'Probabilities': probs_list})
	final_eval_df.to_csv('holdout_preds_bce.csv', sep=' ', float_format='%.2f', header=False, index=False)

	labels = pd.read_csv('/cluster/project1/FFPredLTR/MLDNN/final_evaluation/reference_goa_nk.txt', sep='\t', header=None)
	labels.to_csv('holdout_labels.csv', sep=' ', float_format='%.2f', header=False, index=False)

	file = open('final_evaluation_dict_ltr.pickle', 'rb')
	final_eval_dict = pickle.load(file)
	file.close()

	probs_df = final_eval_dict['probs']
	protein_names_list = []
	go_term_names_list = []
	probs_list = []

	for protein_idx, protein in enumerate(list(probs_df.index)):
		for go_term_idx, go_term in enumerate(list(probs_df.columns)):
			if probs_df.iloc[protein_idx, go_term_idx] >= 0.01:
				protein_names_list.append(protein)
				mod_term = list(go_term)
				mod_term.insert(2, ':')
				mod_term = ''.join(mod_term)
				go_term_names_list.append(mod_term)
				probs_list.append(probs_df.iloc[protein_idx, go_term_idx])
	final_eval_df = pd.DataFrame(data={'Proteins': protein_names_list, 'GO_terms': go_term_names_list, 'Probabilities': probs_list})
	final_eval_df.to_csv('holdout_preds_ltr.csv', sep=' ', float_format='%.2f', header=False, index=False)

	labels = pd.read_csv('/cluster/project1/FFPredLTR/MLDNN/final_evaluation/reference_goa_nk.txt', sep='\t', header=None)
	labels.to_csv('holdout_labels.csv', sep=' ', float_format='%.2f', header=False, index=False)

	file = open('final_evaluation_dict_ltr_only.pickle', 'rb')
	final_eval_dict = pickle.load(file)
	file.close()

	probs_df = final_eval_dict['probs']
	protein_names_list = []
	go_term_names_list = []
	probs_list = []

	for protein_idx, protein in enumerate(list(probs_df.index)):
		for go_term_idx, go_term in enumerate(list(probs_df.columns)):
			if probs_df.iloc[protein_idx, go_term_idx] >= 0.01:
				protein_names_list.append(protein)
				mod_term = list(go_term)
				mod_term.insert(2, ':')
				mod_term = ''.join(mod_term)
				go_term_names_list.append(mod_term)
				probs_list.append(probs_df.iloc[protein_idx, go_term_idx])
	final_eval_df = pd.DataFrame(data={'Proteins': protein_names_list, 'GO_terms': go_term_names_list, 'Probabilities': probs_list})
	final_eval_df.to_csv('holdout_preds_ltr_only.csv', sep=' ', float_format='%.2f', header=False, index=False)