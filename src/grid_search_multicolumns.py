from __future__ import division
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier as GBC
from sklearn.ensemble import RandomForestClassifier as RF
import cPickle as pickle
import os
import sys

from utils.utilities import  get_scores, do_grid_search
from utils.pipeline_noLatLon import *


'''
Routine that evaluates the model on 20.000 points (splitted on train/cross/validate sets) and adding incremental
time window sizes: From 1800 s = 0.5h, to 86400 s = 24h, averages.

run in the console like:
>> python grid_search_multicolumns.py model_1 RF longliners 

I decided to take away the Lat and Lon features as well as the features related with distance to land.
'''

models_dict = {'model_1': ['course_norm_sin_cos','window_1800'],
'model_2': ['course_norm_sin_cos','window_1800','window_3600'],
'model_3': ['course_norm_sin_cos','window_1800','window_3600','window_10800'],
'model_4': ['course_norm_sin_cos','window_1800','window_3600','window_10800','window_21600'],
'model_5': ['course_norm_sin_cos','window_1800','window_3600','window_10800','window_21600','window_43200'],
'model_6': ['course_norm_sin_cos','window_1800','window_3600','window_10800','window_21600','window_43200','window_86400']}


model_number = sys.argv[1] # I was starting from 0, but that was the name of the file
classifier_name = sys.argv[2] 
gear = sys.argv[3]

col_groups = models_dict[model_number]

path = '../data/labeled'
# import pdb, sys
# pdb.set_trace()

'''
possible_gear_types = 
['longliners','trawlers','purse_seines']
possible_model_names = 
['RF','GBC']
'''

full_model_name = model_number + '_' + classifier_name + '_' + gear
X_train, y_train, X_cross, y_cross, X_test, y_test, cols = run_pipeline(path,gear,col_groups)

N_cols = len(cols)  # has a minimum of 3: 'course','measure_daylight','speed'
sqr_N_cols = int(np.sqrt(N_cols))  # maximum value of features to split on
min_N_cols = int(sqr_N_cols/3) #1 if N_cols < 5 else 10 

if N_cols < 9:
	max_features_rf = [3]
else:
	max_features_rf = [min_N_cols,2*min_N_cols,sqr_N_cols]

if classifier_name == 'RF':
	est = RF()
	param_grid = {'n_estimators': [20,50], 
			'max_features': [min_N_cols,2*min_N_cols,sqr_N_cols], 
			'min_samples_split': [3, 10, 15]}
elif classifier_name == 'GBC':
	est = GBC()
	param_grid = {'learning_rate': [0.05, 0.1, 0.2],
			'max_depth': [2, 5],
			'min_samples_leaf': [3, 10, 20]
			}

best_train_score, best_cross_score, best_params = do_grid_search(est,param_grid,X_train,y_train,X_cross,y_cross)


best_classifier = est.set_params(**best_params)
best_classifier.fit(X_train,y_train)
# train_model(best_classifier,X_train,y_train,full_model_name)
a1,r1,f1 = get_scores(best_classifier, X_train, y_train)
a2,r2,f2 = get_scores(best_classifier, X_cross, y_cross)

if 'feature_importances_' in dir(best_classifier):
	feature_importances = np.argsort(best_classifier.feature_importances_)
	if len(cols) < 10:
		important_features = cols[feature_importances[-1:-(len(cols)+1):-1]]
	else:
		important_features = cols[feature_importances[-1:-11:-1]]

pickle_name = '../results/' + full_model_name + '.pkl'
with open(pickle_name, 'w') as pklf:
	pickle.dump(best_classifier, pklf)

output_file = '../results/' + full_model_name + '.txt'

with open(output_file,'wb') as f:

	f.write(model_number + '\n')
	f.write(classifier_name + '\n')
	f.write(gear + '\n')
	f.write(' + '.join(col_groups) + '\n\n')

	f.write("="*50 + '\n\n')
	f.write("Best parameters for {} are: \n {}".format(full_model_name,best_params) + '\n\n')
	f.write("="*50 + '\n')
	f.write("(data) | Accuracy | Recall | F1-Score |" + '\n')
	f.write("-"*50 + '\n')
	f.write("train | {0:.5f} | {1:.5f} | {2:.5f} |".format(a1,r1,f1) + '\n')
	f.write("cross | {0:.5f} | {1:.5f} | {2:.5f} |".format(a2,r2,f2) + '\n')
	f.write("="*50 + '\n\n')
	f.write("top features for best classifier:\n")
	for item in important_features:
		f.write(str(item) +'\n')			




