import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import	defaultdict
import pickle
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
plt.switch_backend('agg')

file = open('branch_list.pickle', 'rb')
list_of_branches = pickle.load(file)
file.close()

font = 15

for margin in ['00', '05', '10']:
	domain = defaultdict(lambda: 0)
	go_names = []
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

		# get the go term names
		go_names.append(branch.go_term)

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
	data = {'bce': fmax_bce_list, 'ltr': fmax_ltr_list, 'ltr_only': fmax_ltr_only_list}
	pvalue_df = pd.DataFrame(data=data)
	pvalue_df.to_csv(margin + '_comparison.csv', index=False, sep='\t')

	x = np.linspace(0.2, 1.0, num=100, endpoint=True)
	y = np.linspace(0.2, 1.0, num=100, endpoint=True)

	# bce vs ltr scatter plot
	plt.figure()
	bce_counter = 0
	ltr_counter = 0
	ltr_only_counter = 0
	for name_idx, (i, j) in enumerate(zip(fmax_bce_list, fmax_ltr_list)):
		if i < j:
			plt.scatter(i, j, marker= 'o', s=20*2**0, c='r')
			ltr_counter += 1
			domain[list_of_branches[name_idx].category + 'ltr' + margin] += 1
		else: 
			plt.scatter(i, j, marker= 'o', s=20*2**0, c='b')
			bce_counter += 1
			domain[list_of_branches[name_idx].category + 'bce' + margin] += 1
	diagonal, = plt.plot(x, y, 'y--', label='Diagonal')
	plt.title('BCE vs BCE+LTR (margin = ' + '.'.join(margin) + ') (All Domains)')
	plt.xlabel('BCE Fmax',fontsize=font)
	plt.ylabel('BCE+LTR Fmax',fontsize=font)
	red_patch = mpatches.Patch(color='red', label='Red Count = %.0f' %(ltr_counter))
	blue_patch = mpatches.Patch(color='blue', label='Blue Count = %.0f' %(bce_counter))
	plt.legend(handles=[red_patch, blue_patch, diagonal])
	plt.savefig('/cluster/project1/FFPredLTR/MLDNN/Plots/PresentationPlots/fmax_comparison_bceltr_margin_' + margin + '_scatter.png')

	plt.figure()
	ind = np.arange(3)
	width = 0.2
	fig, ax = plt.subplots()
	ax.bar(ind, [domain['Molecular Function' + 'ltr' + margin], domain['Biological Process' + 'ltr' + margin], domain['Cellular Component' + 'ltr' + margin]] , width, color='r', label='LTR+BCE', align='center')
	ax.bar(ind + width,  [domain['Molecular Function' + 'bce' + margin], domain['Biological Process' + 'bce' + margin], domain['Cellular Component' + 'bce' + margin]], width, color='b', label='BCE', align='center')
	ax.set_ylabel('Number of Branches',fontsize=font)
	ax.set_title('BCE vs BCE+LTR (margin = ' + '.'.join(margin) + ') (Per Domain)')
	ax.set_xticks(0.5*(ind+ind+width))
	ax.set_xticklabels(['MF', 'BP', 'CC'], fontsize=font)
	ax.legend()
	plt.savefig('/cluster/project1/FFPredLTR/MLDNN/Plots/PresentationPlots/fmax_comparison_ltrbce_per_domain_margin_' + margin + '_barchart.png')

	domain = defaultdict(lambda: 0)

	plt.figure()
	bce_counter = 0
	ltr_counter = 0
	ltr_only_counter = 0
	for name_idx, (i, j) in enumerate(zip(fmax_bce_list, fmax_ltr_only_list)):
		if i < j:
			plt.scatter(i, j, marker= 'o', s=20*2**0, c='g')
			ltr_only_counter += 1
			domain[list_of_branches[name_idx].category + 'ltr' + margin] += 1
		else: 
			plt.scatter(i, j, marker= 'o', s=20*2**0, c='b')
			bce_counter += 1
			domain[list_of_branches[name_idx].category + 'bce' + margin] += 1
	diagonal, = plt.plot(x, y, 'y--', label='Diagonal')
	plt.title('BCE vs LTR (margin = ' + '.'.join(margin) + ') (All Domains)')
	plt.xlabel('BCE Fmax',fontsize=font)
	plt.ylabel('LTR Fmax',fontsize=font)
	red_patch = mpatches.Patch(color='green', label='Green Count = %.0f' %(ltr_only_counter))
	blue_patch = mpatches.Patch(color='blue', label='Blue Count = %.0f' %(bce_counter))
	plt.legend(handles=[red_patch, blue_patch, diagonal])
	plt.savefig('/cluster/project1/FFPredLTR/MLDNN/Plots/PresentationPlots/fmax_comparison_bceltronly_margin_' + margin + '_scatter.png')


	plt.figure()
	ind = np.arange(3)
	width = 0.2
	fig, ax = plt.subplots()
	ax.bar(ind, [domain['Molecular Function' + 'ltr' + margin], domain['Biological Process' + 'ltr' + margin], domain['Cellular Component' + 'ltr' + margin]] , width, color='g', label='LTR', align='center')
	ax.bar(ind + width,  [domain['Molecular Function' + 'bce' + margin], domain['Biological Process' + 'bce' + margin], domain['Cellular Component' + 'bce' + margin]], width, color='b', label='BCE', align='center')
	ax.set_ylabel('Number of Branches', fontsize=font)
	ax.set_title('BCE vs LTR (margin = ' + '.'.join(margin) + ') (Per Domain)')
	ax.set_xticks(0.5*(ind+ind+width))
	ax.set_xticklabels(['MF', 'BP', 'CC'], fontsize=font)
	ax.legend()
	plt.savefig('/cluster/project1/FFPredLTR/MLDNN/Plots/PresentationPlots/fmax_comparison_bceltronly_per_domain_margin_' + margin + '_barchart.png')


	domain = defaultdict(lambda: 0)

	plt.figure()
	bce_counter = 0
	ltr_counter = 0
	ltr_only_counter = 0
	for name_idx, (i, j) in enumerate(zip(fmax_ltr_list, fmax_ltr_only_list)):
		if i <= j:
			plt.scatter(i, j, marker= 'o', s=20*2**0, c='g')
			ltr_only_counter += 1
			domain[list_of_branches[name_idx].category + 'bce' + margin] += 1
		else: 
			plt.scatter(i, j, marker= 'o', s=20*2**0, c='r')
			ltr_counter += 1
			domain[list_of_branches[name_idx].category + 'ltr' + margin] += 1
	diagonal, = plt.plot(x, y, 'y--', label='Diagonal')
	plt.title('BCE+LTR vs LTR (margin = ' + '.'.join(margin) + ') (All Domains)')
	plt.xlabel('BCE+LTR Fmax',fontsize=font)
	plt.ylabel('LTR Fmax',fontsize=font)
	red_patch = mpatches.Patch(color='green', label='Green Count = %.0f' %(ltr_only_counter))
	blue_patch = mpatches.Patch(color='red', label='Red Count = %.0f' %(ltr_counter))
	plt.legend(handles=[red_patch, blue_patch, diagonal])
	plt.savefig('/cluster/project1/FFPredLTR/MLDNN/Plots/PresentationPlots/fmax_comparison_ltrltronly_margin_' + margin + '_scatter.png')


	plt.figure()
	ind = np.arange(3)
	width = 0.2
	fig, ax = plt.subplots()
	ax.bar(ind, [domain['Molecular Function' + 'ltr' + margin], domain['Biological Process' + 'ltr' + margin], domain['Cellular Component' + 'ltr' + margin]] , width, color='r', label='LTR+BCE', align='center')
	ax.bar(ind + width,  [domain['Molecular Function' + 'bce' + margin], domain['Biological Process' + 'bce' + margin], domain['Cellular Component' + 'bce' + margin]], width, color='g', label='LTR', align='center')
	ax.set_ylabel('Number of Branches',fontsize=font)
	ax.set_title('BCE+LTR vs LTR (margin = ' + '.'.join(margin) + ') (Per Domain)')
	ax.set_xticks(0.5*(ind+ind+width))
	ax.set_xticklabels(['MF', 'BP', 'CC'], fontsize=font)
	ax.legend()
	plt.savefig('/cluster/project1/FFPredLTR/MLDNN/Plots/PresentationPlots/fmax_comparison_ltrltronly_per_domain_margin_' + margin + '_barchart.png')
