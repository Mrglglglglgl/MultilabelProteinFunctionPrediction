
"""
A script that creates the csv files containing the Fmax probabilities
for each branch when the BCE loss function is used

"""

import sys
sys.path.insert(0, '/cluster/project1/FFPredLTR/MLDNN/custom_modules')
import numpy as np
import pandas as pd
import pickle

file = open('branches_metrics_BCE.pickle', 'rb')
branch_list = pickle.load(file)
for index, branch in enumerate(branch_list):
	preds_list = []
	labels_list = []
	df = pd.DataFrame(index=branch.x_test.iloc[:,0], columns=branch.label_names, data= (np.round(branch.test_probabilities,2)))
	for counter, ind in enumerate(df.index):
		for counter_2, term in enumerate(df.columns):
			mod_term = list(term)
			mod_term.insert(2, ':')
			mod_term = ''.join(mod_term)
			preds_list.append([ind, mod_term, df.iloc[counter][counter_2]])

	preds_df = pd.DataFrame(index=range(len(branch.x_test.iloc[:,0])*len(branch.label_names)), columns=['Protein', 'GO_term', 'Prediction'], data=preds_list)
	preds_df = preds_df.loc[preds_df['Prediction'] >= 0.01]
	new_index = preds_df.index
	preds_df.set_index('Protein', inplace=True)

	preds_df.to_csv(str(branch.go_term) + '_preds_BCE.csv', sep=' ', float_format='%.2f', header=False)

	df = branch.y_test
	df.set_index('Unnamed: 0', inplace=True)

	for counter, ind in enumerate(df.index):
		for counter_2, term in enumerate(df.columns):
			mod_term = list(term)
			mod_term.insert(2, ':')
			mod_term = ''.join(mod_term)
			labels_list.append([ind, mod_term, df.iloc[counter][counter_2]])

	labels_df= pd.DataFrame(index=range(len(branch.x_test.iloc[:,0])*len(branch.label_names)), columns=['Protein', 'GO_term', 'Label'], data=labels_list)
	labels_df = labels_df.iloc[new_index]
	labels_df = labels_df.loc[labels_df['Label'] == 1.0]
	labels_df = labels_df.drop(columns=['Label'])
	labels_df.set_index('Protein', inplace=True)
	labels_df.to_csv(str(branch.go_term) + '_labels_BCE.csv', sep=' ', float_format='%.2f', header=False)