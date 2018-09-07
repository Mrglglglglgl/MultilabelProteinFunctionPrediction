"""
A script that creates python scripts for parallelized training and hyperparameter 
optimization based on a template 
"""

import re
import pickle

file = open('branch_list.pickle', 'rb')
branch_list = pickle.load(file)
file.close()

cost_function = 'combined'

for branch_number, branch in enumerate(branch_list):
	for parameter_number, parameter in enumerate(range(100),1):
		if cost_function == 'bce':
			with open('main_multilabel_template_bce.py', 'r') as f:
				lines = f.readlines()
				with open('/cluster/project1/FFPredRNN/MLDNN/MLDNN_CPU/' + str(len(branch.label_names)) + '/' + str(branch_number) + '/' + str(parameter_number) + '.py', "w") as g:
					for line in lines:
						line = re.sub(r'branch_number', str(branch_number), line)
						line = re.sub(r'parameter_number', str(parameter_number), line)					
						g.write(line)
		elif cost_function == 'combined':
			with open('main_multilabel_template_combined.py', 'r') as f:
				lines = f.readlines()
				with open('/cluster/project1/FFPredRNN/MLDNN/MLDNN_CPU/' + str(len(branch.label_names)) + '/' + str(branch_number) + '/' + str(parameter_number) + '.py', "w") as g:
					for line in lines:
						line = re.sub(r'branch_number', str(branch_number), line)
						line = re.sub(r'parameter_number', str(parameter_number), line)					
						g.write(line)
		else:
			with open('main_multilabel_template_ltr.py', 'r') as f:
				lines = f.readlines()
				with open('/cluster/project1/FFPredRNN/MLDNN/MLDNN_CPU/' + str(len(branch.label_names)) + '/' + str(branch_number) + '/' + str(parameter_number) + '.py', "w") as g:
					for line in lines:
						line = re.sub(r'branch_number', str(branch_number), line)
						line = re.sub(r'parameter_number', str(parameter_number), line)					
						g.write(line)
