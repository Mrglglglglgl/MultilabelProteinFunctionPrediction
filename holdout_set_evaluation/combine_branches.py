import sys
sys.path.insert(0, '/cluster/project1/FFPredLTR/MLDNN/custom_modules')
import pickle
import combining_tools as ct

file = open('protein_cuda_tensor.pickle', 'rb')
protein_tensor = pickle.load(file)
file.close()

file = open('protein_df.pickle', 'rb')
protein_df = pickle.load(file)
file.close()

file = open('fmax_best_margins_list_ltr_only.pickle', 'rb')
fmax_ltr_only_idx_list = pickle.load(file)
file.close()

file = open('fmax_best_margins_list_ltr.pickle', 'rb')
fmax_ltr_idx_list = pickle.load(file)
file.close()


ct.combine_branches_bce(protein_tensor, protein_df)
ct.final_predictions_bce(protein_tensor, protein_df)

ct.combine_branches_ltr(protein_tensor, protein_df, fmax_ltr_idx_list)
ct.final_predictions_ltr(protein_tensor, protein_df)

ct.combine_branches_ltr_only(protein_tensor, protein_df, fmax_ltr_only_idx_list)
ct.final_predictions_ltr_only(protein_tensor, protein_df)