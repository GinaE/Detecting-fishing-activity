from __future__ import division
# import matplotlib.pyplot as plt
# %matplotlib inline
import numpy as np
import pandas as pd

from sklearn.ensemble import GradientBoostingClassifier as GBC
from sklearn.ensemble import RandomForestClassifier as RF

# We will have the latest version on AWS scikit-learn-0.18.1
# from sklearn.grid_search import GridSearchCV  # sklearn 0.17
from sklearn.model_selection import train_test_split, ParameterGrid
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
	df = df.reset_index()
	df = df.drop('index',axis=1) 
	return df


# def train_model(model,X_train,y_train,model_name):
# 	model.fit(X_train, y_train)
# 	return model #, probabilities

def pickle_model(model,model_name):
	file_name = 'results/{}.pkl'.format(model_name)
	with open(file_name, 'w') as f:
		pickle.dump(model, f)
	pass

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


def do_grid_search(est,param_grid,X_train,y_train,X_cross,y_cross):

	# Initalize our model here
	# est = GBC()
	# param_grid_GBC = {'learning_rate': [0.05, 0.1, 0.2],
	# 			'max_depth': [2, 5, 10],
	# 			'min_samples_leaf': [3, 10, 20]
	# 			}

	# sklearn.model_selection.ParameterGrid(param_grid)
	# predict_proba(X)	

	best_score = 0.5
	for g in ParameterGrid(param_grid):
		print 'running: ', str(g)	# more verbose
		est.set_params(**g)
		est.fit(X_train,y_train)
		cross_score = est.score(X_cross,y_cross) # mean accuracy
		train_score = est.score(X_train,y_train) # mean accuracy
		if cross_score > best_score:    
			best_cross_score = cross_score
			best_train_score = train_score
			best_params = g

	# gs_cv = GridSearchCV(est, param_grid, cv=2, n_jobs=-1, verbose=2).fit(X_train, y_train)
	# return gs_cv.best_score_, gs_cv.best_params_

	return best_train_score, best_cross_score, best_params


def get_scores(fitted_classifier, X_test, y_test):
	# model = classifier(**kwargs)
	# model.fit(X_train, y_train)
	model = fitted_classifier
	y_predict = model.predict(X_test)
	C = confusion_matrix(y_test, y_predict)
	# confusion_matrix(y_true, y_pred)
	# true negatives is C_{0,0}, 
	# false negatives is C_{1,0}, 
	# true positives is C_{1,1}, 
	# false positives is C_{0,1}.
	# P = C_{1,1} + C_{1,0}
	accurary = (C[0][0]+C[1][1]) / len(y_test)
	precision = C[1][1] / (C[0][1] + C[1][1]) 
	recall = C[1][1] / (C[1][1] + C[1][0])
	F1 = 2 * (precision * recall) / (precision + recall)

	# Accuracy, Recall, F1-score
	# return model.score(X_test, y_test), recall, F1
	return accurary, recall, F1

# ==== From here on, modified from SkyTruth's repo =========

def is_fishy(x):
	return x['is_fishing'] == 1

def fishy(x):
	return x[is_fishy(x)]

def nonfishy(x):
	return x[~is_fishy(x)]


def _subsample_even(x0, mmsi, n):
	"""Return `n` subsamples from `x0`

	- all samples have given `mmsi`

	- samples are evenly divided between fishing and nonfishing
	"""
	# Create a mask that is true whenever mmsi is one of the mmsi
	# passed in
	mask = np.zeros([len(x0)], dtype=bool)
	for m in mmsi:
		mask |= (x0['mmsi'] == m) 
	x = x0[mask]  # this makes is a np array?? nope...

	# Pick half the values from fishy rows and half from nonfishy rows.
	f = fishy(x)
	nf = nonfishy(x)
	if n//2 > len(f) or n//2 > len(nf):
		warnings.warn("insufficient items to sample, returning fewer")
	f_index = np.random.choice(f.index, min(n//2, len(f)), replace=False)
	nf_index = np.random.choice(nf.index, min(n//2, len(nf)), replace=False)

	f = f.ix[f_index]
	nf = nf.ix[nf_index]

	# nf = np.random.choice(nf, min(n//2, len(nf)), replace=False)
	ss = pd.concat([f, nf])  #this was making it a np array! yes 
	# np.random.shuffle(ss) # no shuffling
	return ss

def _subsample_proportional(x0, mmsi, n):
	"""Return `n` subsamples from `x0`

	- all samples have given `mmsi`

	- samples are random, so should have ~same be in the same proportions
	  as the x0 for the given mmsi.
	"""
	# Create a mask that is true whenever mmsi is one of the mmsi
	# passed in
	mask = np.zeros([len(x0)], dtype=bool)
	for m in mmsi:
		mask |= (x0['mmsi'] == m)
	x = x0[mask]

	# Pick values randomly
	# Pick values randomly
	
	# ====DEBUGGER=======
	# import pdb
	# pdb.set_trace()

	if n > len(x):
		warnings.warn("Warning, inufficient items to sample, returning {}".format(len(x)))
		n = len(x)
	x_index = np.random.choice(x.index, n, replace=False)
	# ss = np.random.choice(x, n, replace=False)
	ss = x.ix[x_index]
	# np.random.shuffle(ss) # the shuffeling is giving me trouble 
	return ss

 
def sample_by_vessel(x, size = 20000, even_split=None, seed=4321):

	# def load_dataset_by_vessel(path, size = 20000, even_split=None, seed=4321):
	"""Load a dataset from `path` and return train, valid and test sets

	path - path to the dataset
	size - number of samples to return in total, divided between the
		   three sets as (size//2, size//4, size//4)
	even_split - if True, use 50/50 fishing/nonfishing split for training
				  data, otherwise sample the data randomly.

	The data at path is first randomly divided by divided into
	training (1/2), validation (1/4) and test(1/4) data sets.
	These sets are chosen so that MMSI values are not shared
	across the datasets.

	The validation and test data are sampled randomly to get the
	requisite number of points. The training set is sampled randomly
	if `even_split` is False, otherwise it is chose so that half the
	points are fishing.

	"""
	# Set the seed so that we can reproduce results consistently
	np.random.seed(seed)

	# # Load the dataset and strip out any points that aren't classified
	# # (has'is_fishing == Inf)
	# x = np.load(path)['x']
	# x = x[~np.isinf(x['is_fishing']) & ~np.isnan(x['is_fishing']) & ~np.isnan(x['timestamp']) & ~np.isnan(x['speed']) & ~np.isnan(x['course'])]

	if size > len(x):
		print "Warning, insufficient items to sample, returning all"
		size = len(x)

	# Get the list of MMSI and shuffle them. The compute the cumulative
	# lengths so that we can divide the points ~ evenly. Use search
	# sorted to find the division points
	mmsi = list(set(x['mmsi']))
	if even_split is None:
		even_split = x['is_fishing'].sum() > 1 and x['is_fishing'].sum() < len(x)
	if even_split:
		base_mmsi = mmsi
		# Exclude mmsi that don't have at least one fishing or nonfishing point
		mmsi = []
		for m in base_mmsi:
			subset = x[x['mmsi'] == m]
			fishing_count = subset['is_fishing'].sum()
			if fishing_count == 0 or fishing_count == len(subset):
				continue
			mmsi.append(m)
	np.random.shuffle(mmsi)
	nx = len(x)
	sums = np.cumsum([(x['mmsi'] == m).sum() for m in mmsi])
	n1, n2 = np.searchsorted(sums, [nx//2, 3*nx//4])
	if n2 == n1:
		n2 += 1

	# ====DEBUGGER=======
	# import pdb
	# pdb.set_trace()

	train_subsample = _subsample_even if even_split else _subsample_proportional

# try:
	xtrain = train_subsample(x, mmsi[:n1], size//2)
	# xtrain = _subsample_proportional(x, mmsi[:n1], size//2)
	xcross = _subsample_proportional(x, mmsi[n1:n2], size//4)
	xtest = _subsample_proportional(x, mmsi[n2:], size//4)
	# except Exception, e:
	#     print "==== Broken data in the DataFrame ===="
	#     import pdb, sys
	#     sys.last_traceback = sys.exc_info()[2]
	#     pdb.set_trace()

	return xtrain, xcross, xtest


