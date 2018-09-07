
"""
a script that creates the precision recall plots of all three losses for the three domains
"""

import matplotlib.pyplot as plt
import pickle
import numpy as np
import pandas as pd
plt.switch_backend('agg')

font = 15

df_bce = pd.read_csv('/cluster/project1/FFPredLTR/fmaxHoldoutSetEvaluation/holdout_preds_bce.csv.pr_results', skiprows=4)
df_ltr = pd.read_csv('/cluster/project1/FFPredLTR/fmaxHoldoutSetEvaluation//holdout_preds_ltr.csv.pr_results', skiprows=4)
df_ltr_only = pd.read_csv('/cluster/project1/FFPredLTR/fmaxHoldoutSetEvaluation//holdout_preds_ltr_only.csv.pr_results', skiprows=4)

for domain, domain_name in zip(['P','F', 'C'], ['BP Domain','MF Domain', 'CC Domain']):

	recall_bce = list(df_bce[df_bce.category == domain]['avrec'])[::-1]
	precision_bce = list(df_bce[df_bce.category == domain]['avprec'])[::-1]
	f1_bce = list(df_bce[df_bce.category == domain]['F1'])
	best_f1_index_bce = np.argmax(f1_bce)
	best_f1_bce = np.max(f1_bce)
	best_threshold_bce = df_bce[df_bce.category == domain]['threshold'].iloc[best_f1_index_bce]

	recall_ltr = list(df_ltr[df_ltr.category == domain]['avrec'])[::-1]
	precision_ltr = list(df_ltr[df_ltr.category == domain]['avprec'])[::-1]
	f1_ltr = list(df_ltr[df_ltr.category == domain]['F1'])
	best_f1_index_ltr = np.argmax(f1_ltr)
	best_f1_ltr = np.max(f1_ltr)
	best_threshold_ltr = df_ltr[df_ltr.category == domain]['threshold'].iloc[best_f1_index_ltr]

	plt.figure()
	color = 'blue'
	plt.plot(recall_bce, precision_bce, label='BCE Loss', c=color)
	plt.scatter(recall_bce[best_f1_index_bce], precision_bce[best_f1_index_bce], c=color,marker= '>', label = 'Fmax BCE = %.5f' % (best_f1_bce))
	color = 'red'
	plt.plot(recall_ltr, precision_ltr, label='BCE + LTR Loss (combined margins)', c=color)
	plt.scatter(recall_ltr[best_f1_index_ltr], precision_ltr[best_f1_index_ltr] ,c=color,marker= '>', label = 'Fmax BCE+LTR = %.5f' % (best_f1_ltr))

	plt.title('PR Curve, BCE vs BCE+LTR (combined margins) ' + domain_name ,fontsize=font)
	plt.xlabel('Recall', fontsize=font)
	plt.ylabel('Precision', fontsize=font)
	plt.legend()
	plt.savefig('/cluster/project1/FFPredLTR/MLDNN/Plots/PRPlots/bce_ltr' + domain_name)


for domain, domain_name in zip(['P','F', 'C'], ['BP Domain','MF Domain', 'CC Domain']):

	recall_bce = list(df_bce[df_bce.category == domain]['avrec'])[::-1]
	precision_bce = list(df_bce[df_bce.category == domain]['avprec'])[::-1]
	f1_bce = list(df_bce[df_bce.category == domain]['F1'])
	best_f1_index_bce = np.argmax(f1_bce)
	best_f1_bce = np.max(f1_bce)
	best_threshold_bce = df_bce[df_bce.category == domain]['threshold'].iloc[best_f1_index_bce]

	recall_ltr_only = list(df_ltr_only[df_ltr_only.category == domain]['avrec'])[::-1]
	precision_ltr_only = list(df_ltr_only[df_ltr_only.category == domain]['avprec'])[::-1]
	f1_ltr_only = list(df_ltr_only[df_ltr_only.category == domain]['F1'])
	best_f1_index_ltr_only = np.argmax(f1_ltr_only)
	best_f1_ltr_only = np.max(f1_ltr_only)
	best_threshold_ltr_only = df_ltr_only[df_ltr_only.category == domain]['threshold'].iloc[best_f1_index_ltr_only]

	plt.figure()
	color = 'blue'
	plt.plot(recall_bce, precision_bce, label='BCE Loss', c=color)
	plt.scatter(recall_bce[best_f1_index_bce], precision_bce[best_f1_index_bce], c=color,marker= '>', label = 'Fmax BCE = %.5f' % (best_f1_bce))
	color = 'green'
	plt.plot(recall_ltr_only, precision_ltr_only, label='LTR Loss (combined margins)', c=color)
	plt.scatter(recall_ltr_only[best_f1_index_ltr_only], precision_ltr_only[best_f1_index_ltr_only] ,c=color,marker= '>', label = 'Fmax LTR = %.5f' % (best_f1_ltr_only))

	plt.title('PR Curve, BCE vs LTR (combined margins), ' + domain_name,fontsize=font )
	plt.xlabel('Recall', fontsize=font)
	plt.ylabel('Precision', fontsize=font)
	plt.legend()
	plt.savefig('/cluster/project1/FFPredLTR/MLDNN/Plots/PRPlots/bce_ltr_only' + domain_name)



for domain, domain_name in zip(['P','F', 'C'], ['BP Domain','MF Domain', 'CC Domain']):

	recall_ltr = list(df_ltr[df_ltr.category == domain]['avrec'])[::-1]
	precision_ltr = list(df_ltr[df_ltr.category == domain]['avprec'])[::-1]
	f1_ltr = list(df_ltr[df_ltr.category == domain]['F1'])
	best_f1_index_ltr = np.argmax(f1_ltr)
	best_f1_ltr = np.max(f1_ltr)
	best_threshold_ltr = df_ltr[df_ltr.category == domain]['threshold'].iloc[best_f1_index_ltr]

	recall_ltr_only = list(df_ltr_only[df_ltr_only.category == domain]['avrec'])[::-1]
	precision_ltr_only = list(df_ltr_only[df_ltr_only.category == domain]['avprec'])[::-1]
	f1_ltr_only = list(df_ltr_only[df_ltr_only.category == domain]['F1'])
	best_f1_index_ltr_only = np.argmax(f1_ltr_only)
	best_f1_ltr_only = np.max(f1_ltr_only)
	best_threshold_ltr_only = df_ltr_only[df_ltr_only.category == domain]['threshold'].iloc[best_f1_index_ltr_only]

	plt.figure()
	color = 'red'
	plt.plot(recall_ltr, precision_ltr, label='LTR + BCE Loss', c=color)
	plt.scatter(recall_ltr[best_f1_index_ltr], precision_ltr[best_f1_index_ltr], c=color,marker= '>', label = 'Fmax LTR+BCE = %.5f' % (best_f1_ltr))
	color = 'green'
	plt.plot(recall_ltr_only, precision_ltr_only, label='LTR Loss (combined margins)', c=color)
	plt.scatter(recall_ltr_only[best_f1_index_ltr_only], precision_ltr_only[best_f1_index_ltr_only] ,c=color,marker= '>', label = 'Fmax LTR = %.5f' % (best_f1_ltr_only))

	plt.title('PR Curve, LTR vs BCE+LTR (combined margins), ' + domain_name,fontsize=font)
	plt.xlabel('Recall', fontsize=font)
	plt.ylabel('Precision', fontsize=font)
	plt.legend()
	plt.savefig('/cluster/project1/FFPredLTR/MLDNN/Plots/PRPlots/ltr_ltr_only' + domain_name)