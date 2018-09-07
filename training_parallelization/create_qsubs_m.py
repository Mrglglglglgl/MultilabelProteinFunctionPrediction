"""
a script that creates shell scripts to run any failed-to-complete python files in parallel for different hyperparameters
"""
import re
import pickle

file = open('branch_list.pickle', 'rb')
branch_list = pickle.load(file)
file.close()

for branch_number, branch in enumerate(branch_list):
	with open('ArrayJobBranchTemplatem.sh', 'r') as f:
		lines = f.readlines()
		for line in lines:
			line = re.sub(r'branch_number', str(branch_number), line)
			line = re.sub(r'num_go_terms', str(len(branch.label_names)), line)
			with open('/cluster/project1/FFPredRNN/MLDNN/MLDNN_CPU/' + str(len(branch.label_names)) + '/' + str(branch_number) + '/' + 'ArrayJobBranchm' + str(len(branch.label_names)) + str(branch_number) + '.sh', "a") as g:
				g.write(line)