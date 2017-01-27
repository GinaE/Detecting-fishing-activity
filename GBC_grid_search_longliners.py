from __future__ import division
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier as GBC
from sklearn.model_selection import GridSearchCV, train_test_split
from utilities import get_all_data, get_group, do_grid_search_GBC, get_scores, train_model

# gear_type = ['longliners','trawlers','purse_seines']
gear_type = 'longliners'
mod_name = 'GBC' + '_' + gear_type

data_dict = get_all_data('data/labeled')
df = get_group(data_dict,gear_type)

y = df['is_fishing'].astype(int).values
X = df.drop(['mmsi','is_fishing'],axis=1).values
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)

# Grid Search
# param_grid = {'learning_rate': [0.05, 0.1, 0.15, 0.2],
# 				'max_depth': [2, 5, 6, 7, 8, 10],
# 				'min_samples_leaf': [3, 10, 15, 20]
# 				}

param_grid = {'learning_rate': [0.05, 0.1, 0.2],
				'max_depth': [2, 5, 7, 10],
				'min_samples_leaf': [3, 10, 20]
				}



print "RESULTS FOR {}".format(gear_type)
best_score, best_params = do_grid_search_GBC(param_grid,X_train,y_train)
print "="*50
print "The best parameters for {} are: {}".format(mod_name,best_params)
print "Training accuracy =  {}".format(best_score) 

best_classifier = GBC(**best_params)
train_model(best_classifier,X_train,y_train,mod_name)

a1,r1,f1 = get_scores(best_classifier, X_train, y_train)
a2,r2,f2 = get_scores(best_classifier, X_test, y_test)

print "="*50
print "Test scores for {}".format(mod_name)
print "(data) | Accuracy | Recall | F1-Score |"
print "train | {0:.5f} | {0:.5f} | {0:.5f} |".format(a1,r1,f1)
print "test | {0:.5f} | {0:.5f} | {0:.5f} |".format(a2,r2,f2)

if 'feature_importances_' in dir(best_classifier):
	feature_importances = np.argsort(best_classifier.feature_importances_)
	print "top 10 features for best classifier:\n", list(df.columns[feature_importances[-1:-11:-1]])

# probabilities = model.predict_proba(X_val)[:, 1]
# tpr, fpr, thresholds = roc_curve(probabilities, y_val) # takes some time.

# f, a1 = plt.subplots(1, 1, figsize=(20,20))
# a1.plot(fpr,tpr,label=names[i])
# a1.set_xlabel('False Positive Rate')
# a1.set_ylabel('True Positive Rate')
# plt.show()


