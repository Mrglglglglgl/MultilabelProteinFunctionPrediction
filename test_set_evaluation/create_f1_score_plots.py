
"""
a script that creates the scatter plots and barcharts for all losses
regarding results on the test set 
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import	defaultdict
import pickle
import numpy as np
import pandas as pd

plt.switch_backend('agg')

branches_dict = defaultdict()

font = 15
file = open('branch_list.pickle', 'rb')
list_of_branches = pickle.load(file)
file.close()

file = open('/cluster/project1/FFPredLTR/MLDNN/BCE_experiments/branches_metrics_bce_light.pickle', 'rb')
bce_list= pickle.load(file)
file.close()


for margin in ['00', '05', '10']:
	domain = defaultdict(lambda: 0)
	file = open('/cluster/project1/FFPredLTR/MLDNN/LTR_experiments_margin_'+ margin + '/branches_metrics_LTR_margin_'+ margin + '_light.pickle', 'rb')
	ltr_list = pickle.load(file)
	file.close()

	file = open('/cluster/project1/FFPredLTR/MLDNN/LTR_only_experiments_margin_'+ margin + '/branches_metrics_LTR_only_margin_'+ margin + '_light.pickle', 'rb')
	ltr_only_list = pickle.load(file)
	file.close()

	go_names = []
	f1_bce_list = []
	f1_ltr_list = []
	f1_ltr_only_list = []

	for branch_bce, branch_ltr, branch_ltr_only in zip(bce_list, ltr_list, ltr_only_list):
		# Overall
		go_names.append(branch_bce.go_term)
		f1_bce_list.append(branch_bce.f1_median)
		f1_ltr_list.append(branch_ltr.f1_median)
		f1_ltr_only_list.append(branch_ltr_only.f1_median)		
	x = np.linspace(0.05, 0.7, num=100, endpoint=True)
	y = np.linspace(0.05, 0.7, num=100, endpoint=True)

	# bce vs ltr scatter plot
	plt.figure()
	bce_counter = 0
	ltr_counter = 0
	ltr_only_counter = 0
	for name_idx, (i, j) in enumerate(zip(f1_bce_list, f1_ltr_list)):
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
	plt.xlabel('BCE F1',fontsize=font)
	plt.ylabel('BCE+LTR F1',fontsize=font)
	red_patch = mpatches.Patch(color='red', label='Red Count = %.0f' %(ltr_counter))
	blue_patch = mpatches.Patch(color='blue', label='Blue Count = %.0f' %(bce_counter))
	plt.legend(handles=[red_patch, blue_patch, diagonal])
	plt.savefig('/cluster/project1/FFPredLTR/MLDNN/Plots/PresentationPlots/f1_comparison_bceltr_margin_' + margin + '_scatter.png')

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
	plt.savefig('/cluster/project1/FFPredLTR/MLDNN/Plots/PresentationPlots/f1_comparison_ltrbce_per_domain_margin_' + margin + '_barchart.png')

	domain = defaultdict(lambda: 0)

	plt.figure()
	bce_counter = 0
	ltr_counter = 0
	ltr_only_counter = 0
	for name_idx, (i, j) in enumerate(zip(f1_bce_list, f1_ltr_only_list)):
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
	plt.xlabel('BCE F1',fontsize=font)
	plt.ylabel('LTR F1',fontsize=font)
	red_patch = mpatches.Patch(color='green', label='Green Count = %.0f' %(ltr_only_counter))
	blue_patch = mpatches.Patch(color='blue', label='Blue Count = %.0f' %(bce_counter))
	plt.legend(handles=[red_patch, blue_patch, diagonal])
	plt.savefig('/cluster/project1/FFPredLTR/MLDNN/Plots/PresentationPlots/f1_comparison_bceltronly_margin_' + margin + '_scatter.png')


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
	plt.savefig('/cluster/project1/FFPredLTR/MLDNN/Plots/PresentationPlots/f1_comparison_bceltronly_per_domain_margin_' + margin + '_barchart.png')


	domain = defaultdict(lambda: 0)

	plt.figure()
	bce_counter = 0
	ltr_counter = 0
	ltr_only_counter = 0
	for name_idx, (i, j) in enumerate(zip(f1_ltr_list, f1_ltr_only_list)):
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
	plt.xlabel('BCE+LTR F1',fontsize=font)
	plt.ylabel('LTR F1',fontsize=font)
	red_patch = mpatches.Patch(color='green', label='Green Count = %.0f' %(ltr_only_counter))
	blue_patch = mpatches.Patch(color='red', label='Red Count = %.0f' %(ltr_counter))
	plt.legend(handles=[red_patch, blue_patch, diagonal])
	plt.savefig('/cluster/project1/FFPredLTR/MLDNN/Plots/PresentationPlots/f1_comparison_ltrltronly_margin_' + margin + '_scatter.png')


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
	plt.savefig('/cluster/project1/FFPredLTR/MLDNN/Plots/PresentationPlots/f1_comparison_ltrltronly_per_domain_margin_' + margin + '_barchart.png')