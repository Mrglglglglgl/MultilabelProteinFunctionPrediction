"""
a script that searches and saves the model architecture and weights for which
the best f1 in the validation set was achieved using the bce loss
"""

import pickle
from shutil import copyfile


file = open('/cluster/project1/FFPredRNN/MLDNN/branch_list.pickle', 'rb')
branch_list = pickle.load(file)
file.close()
sample = -1

margin = '00'

for branch_number, branch in enumerate(branch_list):
	f1_benchmark = -1
	for index, dummy in enumerate(range(100),1):
		path =  '/cluster/project1/FFPredRNN/Log_files/' + str(len(branch.label_names)) + '_log_file/' + str(branch_number) + '/' + str(branch.go_term) + str(index) + '_trained_LTR.pickle'
		file = open(path, 'rb')
		trained_branch = pickle.load(file)[0]
		file.close()
		if trained_branch.best_f1 > f1_benchmark:
			f1_benchmark = trained_branch.best_f1
			sample = index
			go_term = trained_branch.go_term
			best_branch = trained_branch
			pt_path = '/cluster/project1/FFPredRNN/Log_files/' + str(len(branch.label_names)) + '_log_file/' + str(branch_number) + '/' + str(branch.go_term) + 'LTR' + str(index) + '.pt'
	copyfile(pt_path, '/cluster/project1/FFPredRNN/MLDNN/' + str(branch.go_term) + 'bce_margin_' + margin + '.pt')
	file = open('/cluster/project1/FFPredRNN/MLDNN/' + str(branch.go_term) + 'trained_bce_margin_' + margin + '.pickle', 'wb')
	pickle.dump(best_branch, file)
	file.close()

