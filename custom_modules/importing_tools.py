
"""
a module that sets the paths and creates the dictionaries needed to import the data
"""

import glob
import re
import pandas as pd

def create_paths():

    """
    creates the paths of all the files needed
    """

    all_data_x_path = '/cluster/project1/FFPredLTR/MTDNN/mtdnn_upload/data/all_data_X.csv'
    all_data_y_path = '/cluster/project1/FFPredLTR/MTDNN/mtdnn_upload/data/all_data_Y.csv'
    dataset_paths = {'train':'/cluster/project1/FFPredLTR/MTDNN/mtdnn_upload/data/train_sets/*train.csv',
    'test': '/cluster/project1/FFPredLTR/MTDNN/mtdnn_upload/data/train_sets/*test.csv', 
    'valid': '/cluster/project1/FFPredLTR/MTDNN/mtdnn_upload/data/train_sets/*valid.csv'}

    category_paths = {'Biological Process': '/cluster/project1/FFPredLTR/MTDNN/mtdnn_upload/data/sub/bp_branches.txt',
    'Molecular Function': '/cluster/project1/FFPredLTR/MTDNN/mtdnn_upload/data/sub/mf_branches.txt', 
    'Cellular Component': '/cluster/project1/FFPredLTR/MTDNN/mtdnn_upload/data/sub/cc_branches.txt'}


    return all_data_x_path, all_data_y_path, dataset_paths, category_paths

def create_x_data_dataframe(path, separator):
    """
    creates a dataframe with all the proteins
    
    """

    x_data_dataframe = pd.read_csv(path, sep=separator)
    x_data_dataframe['Unnamed: 0'] = x_data_dataframe['Unnamed: 0'].map(lambda x: str(x)[1:-1])
    
    return x_data_dataframe

def create_y_data_dataframe(path, separator):
    """
    creates a dataframe with all the proteins
    
    """

    y_data_dataframe = pd.read_csv(path, sep=separator)
    y_data_dataframe['Unnamed: 0'] = y_data_dataframe['Unnamed: 0'].map(lambda x: str(x)[1:-1])
    
    return y_data_dataframe


def create_branch_dict(paths, all_proteins_df):
    
    """
    creates a dictionary with all the datasets for each branch
    """
    
    dict_of_branches = defaultdict(dict)
    set_of_labels = []
    for setting, path in paths.items():
        glob.glob('/path/to/dir/')
        files = glob.glob(path)
        for name in files:
            go_name = re.sub(r'.*GO', 'GO', name)
            go_name = re.sub('_train.csv', '', go_name)
            go_name = re.sub('_test.csv', '', go_name)
            go_name = re.sub('_valid.csv', '', go_name)
            with open(name) as f:          
                dict_of_branches[go_name]['y_' + setting] = pd.read_csv(f, sep='\t')               
                dict_of_branches[go_name]['x_' + setting] = all_proteins_df.loc[all_proteins_df['Unnamed: 0'].isin(dict_of_branches[go_name]['y_' + setting]['Unnamed: 0'])]
                dict_of_branches[go_name]['x_' + setting] = dict_of_branches[go_name]['x_' + setting].set_index('Unnamed: 0')
                dict_of_branches[go_name]['x_' + setting] = dict_of_branches[go_name]['x_' + setting].reindex(index=dict_of_branches[go_name]['y_' + setting]['Unnamed: 0'])
                dict_of_branches[go_name]['x_' + setting] = dict_of_branches[go_name]['x_' + setting].reset_index()

    return dict_of_branches


def create_category_dict(paths):
    """
    creates a dictionary with all go terms and their categories
    
    """
    dict_of_categories = defaultdict(str)
    for category, path in paths.items():
        print(category, path)
        glob.glob('/path/to/dir/')
        files = glob.glob(path)
        for name in files:
            with open(name) as f:
                for go_term in f:
                    go_term = go_term.replace(':', '')[:-1]
                    dict_of_categories[go_term] = category
                  
    return dict_of_categories
