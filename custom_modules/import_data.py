
"""
a script that creates pickle files with all the necessary data for all branches and domains
"""

from collections import Counter
import importing_tools as it
from branches import Branch
import pickle
import warnings

warnings.filterwarnings("ignore")

all_proteins_path, all_labels_path, dataset_paths, category_paths = it.create_paths()
all_proteins_df = it.create_x_data_dataframe(all_proteins_path, '\t')
all_labels_df = it.create_y_data_dataframe(all_labels_path, '\t')
datasets_dict = it.create_branch_dict(dataset_paths, all_proteins_df)
categories_dict = it.create_category_dict(category_paths)
branch_list = []
for key in datasets_dict.keys():
    branch_list.append(Branch(key, datasets_dict, categories_dict))

bp_terms_list = []
mf_terms_list = []
cc_terms_list = []
for branch in branch_list:
    if branch.category == 'Biological Process':
        for i in range(len(branch.label_names)):
            bp_terms_list.append(branch.label_names[i])
    elif branch.category == 'Molecular Function':
        for i in range(len(branch.label_names)):
            mf_terms_list.append(branch.label_names[i])
    else: 
        for i in range(len(branch.label_names)):
            cc_terms_list.append(branch.label_names[i])
            
bp_terms_dict = Counter(bp_terms_list)
mf_terms_dict = Counter(mf_terms_list)
cc_terms_dict = Counter(cc_terms_list)

file = open('branch_list.pickle', 'wb')
pickle.dump(branch_list, file)
file.close()

file = open('bp_terms_dict.pickle', 'wb')
pickle.dump(bp_terms_dict, file)
file.close()

file = open('mf_terms_dict.pickle', 'wb')
pickle.dump(mf_terms_dict, file)
file.close()

file = open('cc_terms_dict.pickle', 'wb')
pickle.dump(cc_terms_dict, file)
file.close()