# reconstructing_tracks.py
from __future__ import division
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from mpl_toolkits.basemap import Basemap
import cPickle as pickle
import scipy.stats as st

from utils.pipeline_noLatLon import keep_columns, X_y_split, columns_models_dict
from utils.utilities import get_all_data, get_group



def data_prep(df,mod_num):

	models_dict = {'model_1': ['course_norm_sin_cos','window_1800'],
	'model_2': ['course_norm_sin_cos','window_1800','window_3600'],
	'model_3': ['course_norm_sin_cos','window_1800','window_3600',
		'window_10800'],
	'model_4': ['course_norm_sin_cos','window_1800','window_3600',
		'window_10800','window_21600'],
	'model_5': ['course_norm_sin_cos','window_1800','window_3600',
		'window_10800','window_21600','window_43200'],
	'model_6': ['course_norm_sin_cos','window_1800','window_3600',
		'window_10800','window_21600','window_43200','window_86400']}

	df = df.drop_duplicates()
	col_groups = models_dict[mod_num] # col_groups = models_dict['model_4'] 
	df_to_predict_on = keep_columns(df, col_groups)
	X, y, cols = X_y_split(df_to_predict_on)
	return X, y, cols  # no dataFrames

def make_prediction(X,pickeled_model):
	with open(pickeled_model) as f:
	    model = pickle.load(f)
	prediction = model.predict(X)
	return prediction

def plot_track_predictions(df,mod_num,mmsi,pickeled_model,up,down,left,right):

	# df is for a single vessel, a single mmsi
	# mod_num = 'model_4'
	# pickeled_model = 'final_models/model_3_RF_trawlers.pkl'

	fig_name = '../results/best_reconstructions/' + str(mmsi) +'_transparency_' + pickeled_model[len('../results/final_models/'):-len('.pkl')]+'.png'

	# # ====DEBUGGER=======
	# import pdb, sys
	# pdb.set_trace()

	X, y, cols = data_prep(df,mod_num)
	prediction = make_prediction(X,pickeled_model)

	df_fishing = df[df.is_fishing == 1]
	df_pred_fishing = df[prediction == 1]

	latitude = df['lat']
	longitude = df['lon']
	
	max_lat = max(latitude) + up 
	min_lat = min(latitude) - down
	max_lon = max(longitude) + right
	min_lon = min(longitude) - left

	# for the Africa zoom
	# max_lat = -16
	# min_lat = -25
	# max_lon = 42 # center_x + win
	# min_lon = 32


	plt.figure(figsize=(10,6))

	plt.subplot(1,2,1)
	m = Basemap(projection='cyl', llcrnrlat=min_lat, urcrnrlat=max_lat,llcrnrlon=min_lon, urcrnrlon=max_lon, resolution='h', area_thresh=1000.)
	# m.bluemarble()
	m.drawlsmask(land_color=(1,215/255,0,1), ocean_color='w')
	m.drawcoastlines(linewidth=0.5)
	m.drawcountries(linewidth=0.5)
	# m.drawmapboundary(fill_color='aqua')

	# m.drawparallels(np.arange(10.,35.,5.))
	# m.drawmeridians(np.arange(-120.,-80.,10.))
	
	x, y = m(longitude, latitude)
	x1, y1 = m(df_fishing['lon'], df_fishing['lat'])
	plt.plot(x,y,'r-',alpha=0.6,linewidth=1)
	plt.plot(x1,y1,'go') 
	plt.title('Fishing sites')

	plt.subplot(1,2,2)
	m = Basemap(projection='cyl', llcrnrlat=min_lat, urcrnrlat=max_lat,llcrnrlon=min_lon, urcrnrlon=max_lon, resolution='h', area_thresh=1000.)
	# m.bluemarble()
	m.drawlsmask(land_color=(1,215/255,0,1), ocean_color='w') # color in rgb should be gold
	m.drawcoastlines(linewidth=0.5)
	m.drawcountries(linewidth=0.5)
	# m.drawmapboundary(fill_color='aqua')
	x, y = m(longitude, latitude)
	x1, y1 = m(df_pred_fishing['lon'], df_pred_fishing['lat'])
	plt.plot(x,y,'r-',alpha=0.6,linewidth=1)
	plt.plot(x1,y1,'bo',ms=5) 
	plt.title('Predicted fishing sites') 

	plt.savefig(fig_name)
	# plt.show()

def load_mmsis(gear,data_set):
	mmsis_file_name = './used_mmsi/'+gear+'_mmsi_'+data_set+'.txt'
	mmsis = pd.read_csv(mmsis_file_name)
	return mmsis.mmsi.unique()


if __name__ == '__main__':

	path = '../data/labeled'
	data_dict = get_all_data(path)  # these steps are taking tooooo long.

	gears = ['longliners','trawlers','purse_seines']

	for gear in gears:
		if gear == 'longliners':
			mod_num = 'model_6'
			pickeled_model = '../results/final_models/model_6_RF_longliners.pkl'
			# best_mmsi = 134393118622376  # too small latitude span.
			best_mmsi = 36427802545466 # Around South Africa
			# best_mmsi = 207649577408623  # close to micronesia
			up = 2
			down = 2
			left = 5
			right = 5

		elif gear == 'trawlers':
			mod_num = 'model_3'
			pickeled_model = '../results/final_models/model_3_RF_trawlers.pkl'
			best_mmsi = 229561307305795 # south of Spain
			up = 0.5
			down = 0.5
			left = 0.1
			right = 0.1

		else:
			mod_num = 'model_3'
			pickeled_model = '../results/final_models/model_3_GBC_purse_seines.pkl'
			best_mmsi = 10880510825243 # Japan
			up = 10
			down = 10
			left = 10
			right = 1

		df = get_group(data_dict,gear)

		df_single_vessel = df[df.mmsi==float(best_mmsi)]


		plot_track_predictions(df_single_vessel,mod_num,best_mmsi,pickeled_model,up,down,left,right)


		# # ====DEBUGGER=======
		# import pdb, sys
		# pdb.set_trace()












