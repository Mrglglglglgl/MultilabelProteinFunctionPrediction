"""
a script that creates csv files containing metrics achieved
in the test set for all losses 
"""

import pickle
import pandas as pd

file = open('/cluster/project1/FFPredLTR/MLDNN/BCE_experiments/branches_metrics_light.pickle', 'rb')
metrics = pickle.load(file)
file.close()
lengths_list = []
names_list = []
f1_list = []
recall_list = []
precision_list = []
mcc_list = []
auc_list = []


for counter, branch in enumerate(metrics):
	lengths_list.append(int(len(branch.label_names)))
	names_list.append(branch.go_term)
	f1_list.append(branch.f1_median)
	recall_list.append(branch.recall_median)
	precision_list.append(branch.precision_median)
	mcc_list.append(branch.median_mcc)
	auc_list.append(branch.roc_auc_median)

df = pd.DataFrame(data={'Branch root term': names_list ,'# GO Terms': lengths_list, 'Median F1': f1_list, 'Median MCC': mcc_list, 
	'Median ROC_AUC': auc_list, 'Median Precision': precision_list, 'Median Recall': recall_list})

df.to_csv('medians_bce.csv', sep=' ')


for margin in ['00', '05', '10']:

	file = open('/cluster/project1/FFPredLTR/MLDNN/LTR_experiments_margin_' + margin + '/branches_metrics_LTR_margin_' + margin + '_light.pickle', 'rb')
	metrics = pickle.load(file)
	file.close()
	lengths_list = []
	names_list = []
	f1_list = []
	recall_list = []
	precision_list = []
	mcc_list = []
	auc_list = []
	for counter, branch in enumerate(metrics):
		lengths_list.append(int(len(branch.label_names)))
		names_list.append(branch.go_term)
		f1_list.append(branch.f1_median)
		recall_list.append(branch.recall_median)
		precision_list.append(branch.precision_median)
		mcc_list.append(branch.median_mcc)
		auc_list.append(branch.roc_auc_median)

	df = pd.DataFrame(data={'Branch root term': names_list ,'# GO Terms': lengths_list, 'Median F1': f1_list, 'Median MCC': mcc_list, 
	'Median ROC_AUC': auc_list, 'Median Precision': precision_list, 'Median Recall': recall_list})

	df.to_csv('medians_LTR_margin_' + margin + '.csv', sep=' ')


	file = open('/cluster/project1/FFPredLTR/MLDNN/LTR_only_experiments_margin_' + margin + '/branches_metrics_LTR_only_margin_' + margin +'_light.pickle', 'rb')
	metrics = pickle.load(file)
	file.close()
	lengths_list = []
	names_list = []
	f1_list = []
	recall_list = []
	precision_list = []
	mcc_list = []
	auc_list = []
	for counter, branch in enumerate(metrics):
		lengths_list.append(int(len(branch.label_names)))
		names_list.append(branch.go_term)
		f1_list.append(branch.f1_median)
		recall_list.append(branch.recall_median)
		precision_list.append(branch.precision_median)
		mcc_list.append(branch.median_mcc)
		auc_list.append(branch.roc_auc_median)

	df = pd.DataFrame(data={'Branch ID': names_list ,'No. GO Terms': lengths_list, 'Median F1': f1_list, 'Median MCC': mcc_list, 
		'Median ROC_AUC': auc_list, 'Median Precision': precision_list, 'Median Recall': recall_list})

	df.to_csv('medians_LTR_only_margin_' + margin + '.csv', sep=' ', index=False)