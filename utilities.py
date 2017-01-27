from __future__ import division
# import matplotlib.pyplot as plt
# %matplotlib inline
import numpy as np
import pandas as pd

from sklearn.ensemble import GradientBoostingClassifier as GBC
from sklearn.ensemble import RandomForestClassifier as RF

# We will have the latest version on AWS scikit-learn-0.18.1
# from sklearn.grid_search import GridSearchCV  # sklearn 0.17
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.metrics import f1_score, confusion_matrix, roc_curve, auc

import cPickle as pickle
import os


def get_data(path):
	'''
	loads the data specified on the path and puts it on a dataframe
	input: path (srt)
	output: dataframe
	'''

	x = np.load(path)['x']
	# changed "classification" by "is_fishing" 
	x = x[~np.isinf(x['is_fishing']) & ~np.isnan(x['is_fishing']) & ~np.isnan(x['timestamp']) & ~np.isnan(x['speed']) & ~np.isnan(x['course'])]
	df = pd.DataFrame(x)
	return df

def get_all_data(dir = None):
	'''
	it will loop over the files in dir and load them
	input: directory name (srt)
	output: dictionary of dataframes with key = name of file
	'''
	if dir is None:
		dir = os.path.join(os.path.dirname(__file__), '.', 'data/labeled')

	datasets = {}
	for filename in os.listdir(dir):
		if filename.endswith('.measures.labels.npz'):
			name = filename[:-len('.measures.labels.npz')]
			#             datasets[name] = dict(zip(['all', 'train', 'cross', 'test'], load_dataset_by_vessel(os.path.join(dir, filename))))
			x = np.load(os.path.join(dir, filename))['x']
			x = x[~np.isinf(x['is_fishing']) & ~np.isnan(x['is_fishing']) & ~np.isnan(x['timestamp']) & ~np.isnan(x['speed']) & ~np.isnan(x['course'])]
			datasets[name] = pd.DataFrame(x)
			# print name
	return datasets

def get_group(data_dict,gear_type):

	# data_dict = get_all_data('data/labeled')

	group_per_gear = {

	'longliners':  
	['alex_crowd_sourced_Drifting_longlines',
	'kristina_longliner_Drifting_longlines',
	'pybossa_project_3_Drifting_longlines'],

	'purse_seines':  
	['alex_crowd_sourced_Purse_seines',
	'kristina_ps_Purse_seines',
	'pybossa_project_3_Purse_seines'],

	'trawlers':  
	['kristina_trawl_Trawlers',
	'pybossa_project_3_Trawlers'],

	'others':
	['alex_crowd_sourced_Unknown',
	'kristina_longliner_Fixed_gear',
	'kristina_longliner_Unknown',
	'kristina_ps_Unknown',
	'pybossa_project_3_Unknown',
	'pybossa_project_3_Pole_and_line',  
	'pybossa_project_3_Trollers'  
	'pybossa_project_3_Fixed_gear'], 

	'false_positives':
	['false_positives_Drifting_longlines',
	'false_positives_Fixed_gear',
	'false_positives_Purse_seines',
	'false_positives_Trawlers',
	'false_positives_Unknown']

	}
	
	df = pd.concat([ data_dict[filename] for filename in group_per_gear[gear_type]]) 
	return df


def train_model(model,X_train,y_train,model_name):
	'''

	'''
	model.fit(X_train, y_train)
	file_name = 'results/{}.pkl'.format(model_name)
	with open(file_name, 'w') as f:
	    pickle.dump(model, f)
	return model #, probabilities


def roc_curve(probabilities, labels):
	'''
	Sort instances by their prediction strength (the probabilities)
	For every instance in increasing order of probability:
	    Set the threshold to be the probability
	    Set everything above the threshold to the positive class
	    Calculate the True Positive Rate (aka sensitivity or recall)
	    Calculate the False Positive Rate (1 - specificity)
	Return three lists: TPRs, FPRs, thresholds
	'''
	prob_order = np.sort(probabilities)    
	TPRs = []
	FPRs = []
	for pr in prob_order:
		th = pr
		classification = probabilities >= th  # boolean       
		TP = sum(np.logical_and(classification,labels))              
		TPRs.append(float(TP))        
		labels_neg = -1*labels + 1
		FP = sum(np.logical_and(classification, labels_neg))       
		FPRs.append(float(FP))  
	P = sum(labels)
	N = len(labels) - P   
	return TPRs/P, FPRs/N, prob_order


def do_grid_search_GBC(param_grid,X_train,y_train):

    # Initalize our model here
	est = GBC()

	# param_grid = {'learning_rate': [0.1, 0.05, 0.02],
	# 				'max_depth': [2, 3],
	# 				'min_samples_leaf': [3, 5],
	# 				}

	gs_cv = GridSearchCV(est, param_grid, n_jobs=-1, verbose=2).fit(X_train, y_train)
	return gs_cv.best_score_, gs_cv.best_params_

def do_grid_search_RF(param_grid,X_train,y_train):

    # Initalize our model here
	est = RF()

	# param_grid = {'n_estimators': [10,20,50,80],
	#              'max_features': [10,50,80],
	#              'min_samples_split': [3, 10, 15, 20]
	#              }

	gs_cv = GridSearchCV(est, param_grid, n_jobs=-1,verbose=2).fit(X_train, y_train)
	return gs_cv.best_score_, gs_cv.best_params_


def get_scores(fitted_classifier, X_test, y_test):
	# model = classifier(**kwargs)
	# model.fit(X_train, y_train)
	model = fitted_classifier
	y_predict = model.predict(X_test)
	confusion_matrix(y_test, y_predict)
	matrix = confusion_matrix(y_test, y_predict)
	precision = matrix[1][1] / (matrix[1][1] + matrix[0][1]) 
	recall = matrix[1][1] / (matrix[1][1] + matrix[1][0])
	F1 = 2 * (precision * recall) / (precision + recall)

	# Accuracy, Recall, F1-score
	return model.score(X_test, y_test), recall, F1
