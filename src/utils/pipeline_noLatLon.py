# pipeline_noLatLon.py
import pandas as pd 
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier as GBC
from sklearn.ensemble import RandomForestClassifier as RF
from utilities import get_all_data, get_group
from utilities import _subsample_even, _subsample_proportional, sample_by_vessel
import csv
import json

def threshold_fishing(df):
	df['is_fishing'] = df['is_fishing'].apply(lambda x: 1 if x>0.5 else 0) 
	return df

def keep_columns(df,col_groups):

	from pipeline_noLatLon import columns_models_dict 
	# minimal model
	cols_to_keep = ['timestamp','mmsi','course','measure_daylight','speed','is_fishing']
 
	if col_groups:

		if col_groups == 'all':
			for key in columns_models_dict:
				cols_to_keep += [col for col in columns_models_dict[key]]
		else:
			for col_g in col_groups:
				cols_to_keep += columns_models_dict[col_g] 
	
	df =  df[cols_to_keep]
	# N_cols = len(cols_to_keep)
	return df

def random_split(df):
	y = df['is_fishing'].astype(int).values
	df_X = df.drop(['mmsi','is_fishing','timestamp'],axis=1)
	X = df_X.values
	X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)
	return X_train, X_test, y_train, y_test

def X_y_split(df):
	y = df['is_fishing'].astype(int).values
	df_X = df.drop(['mmsi','is_fishing','timestamp'],axis=1)
	cols = df_X.columns
	X = df_X.values
	return X, y, cols

def save_mmsi(mmsi_series,file_name):
	with open(file_name, 'w') as outfile:
	    mmsi_series.astype(int).to_csv(outfile,index=False,header=True,encoding='utf-8') 

def run_pipeline(path,gear,col_groups):
	# path = '../data/labeled'
	# gear = 'longliners'

	data_dict = get_all_data(path)
	df = get_group(data_dict,gear)
	df.reindex() 
	df = threshold_fishing(df)

	# decide what group of variables to add to the basic model. It has to be a list!
	df = keep_columns(df, col_groups = col_groups)

	df_train, df_cross, df_test = sample_by_vessel(df, size = 20000, even_split=None, seed=4321)

	mmsi_tr = set(df_train.mmsi)
	mmsi_cr = set(df_cross.mmsi)
	mmsi_te = set(df_test.mmsi)

	# ====DEBUGGER=======
	# import pdb, sys
	# pdb.set_trace()

	# passing the mmsi as pandas series instead of lists... 
	# save_mmsi(df_train.mmsi,'used_mmsi/'+gear+'_mmsi_train.txt')
	# save_mmsi(df_cross.mmsi,'used_mmsi/'+gear+'_mmsi_cross.txt')
	# save_mmsi(df_test.mmsi,'used_mmsi/'+gear+'_mmsi_test.txt')

	X_train, y_train, cols = X_y_split(df_train)
	X_cross, y_cross, cols = X_y_split(df_cross)
	X_test, y_test, cols = X_y_split(df_test)

	return X_train, y_train, X_cross, y_cross, X_test, y_test, cols

models_dict = {'model_1': ['course_norm_sin_cos','window_1800'],
'model_2': ['course_norm_sin_cos','window_1800','window_3600'],
'model_3': ['course_norm_sin_cos','window_1800','window_3600','window_10800'],
'model_4': ['course_norm_sin_cos','window_1800','window_3600','window_10800','window_21600'],
'model_5': ['course_norm_sin_cos','window_1800','window_3600','window_10800','window_21600','window_43200'],
'model_6': ['course_norm_sin_cos','window_1800','window_3600','window_10800','window_21600','window_43200','window_86400']}

columns_models_dict = {

'dist_to_land': [
'distance_from_port',
'distance_from_shore',
'measure_distance_from_port'],

'course_norm_sin_cos' : [
'measure_course',
'measure_cos_course',
'measure_sin_course'],

# 'measure_pos_1800', took this column out for now.... (feb 2nd) I think it was overfitting to this
'window_1800':['measure_coursestddev_1800_log',
'measure_daylightavg_1800',
'measure_speedstddev_1800',
'measure_count_1800',
'measure_courseavg_1800',
'measure_coursestddev_1800',
'measure_speedavg_1800',
'measure_speedstddev_1800_log'],

'window_3600': ['measure_count_3600',
'measure_speedstddev_3600',
'measure_speedavg_3600',
'measure_courseavg_3600',
'measure_daylightavg_3600',
'measure_coursestddev_3600',
'measure_speedstddev_3600_log',
'measure_coursestddev_3600_log'],

'window_10800': ['measure_coursestddev_10800_log',
'measure_speedstddev_10800',
'measure_speedavg_10800',
'measure_daylightavg_10800',
'measure_courseavg_10800',
'measure_count_10800',
'measure_speedstddev_10800_log',
'measure_coursestddev_10800'],

'window_21600': ['measure_coursestddev_21600',
'measure_speedavg_21600',
'measure_count_21600',
'measure_coursestddev_21600_log',
'measure_speedstddev_21600_log',
'measure_speedstddev_21600',
'measure_daylightavg_21600',
'measure_courseavg_21600'],

'window_43200': ['measure_coursestddev_43200',
'measure_courseavg_43200',
'measure_daylightavg_43200',
'measure_coursestddev_43200_log',
'measure_speedavg_43200',
'measure_count_43200',
'measure_speedstddev_43200_log',
'measure_speedstddev_43200'],

'window_86400': ['measure_speedavg_86400',
'measure_count_86400',
'measure_speedstddev_86400_log',
'measure_speedstddev_86400',
'measure_coursestddev_86400_log',
'measure_coursestddev_86400',
'measure_daylightavg_86400',
'measure_courseavg_86400'],

}


# if __name__ == '__main__':

# 	path = '../data/labeled'
# 	gear = 'longliners'

# 	mmsi_tr, mmsi_cr, mmsi_te, X_train, y_train, X_cross, y_cross, X_test, y_test, cols = run_pipeline(path,gear)
