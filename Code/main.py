
# Author Mengwei Yang 

from __future__ import print_function
import os
import xgboost as xgb
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.feature_extraction.text import TfidfTransformer
from xgboost import XGBClassifier
from xgboost import plot_tree
import matplotlib.pyplot as plt
import numpy as np
import datetime as dt
from time import time

# use rabit to achieve communicatins between nodes
xgb.rabit.init()

print('MPI world-size: '+str(xgb.rabit.get_world_size()))
print('MPI get-rank  : '+str(xgb.rabit.get_rank()))
print('MPI hostname  : '+str(xgb.rabit.get_processor_name()))

rank = xgb.rabit.get_rank()

# assign dataset to different node(server and local)
if (rank ==0): 
	df = pd.read_csv('creditcard1_train.csv')
	X_train = df[df.columns[:-1].tolist()]
	y_train = df[df.columns[-1]]

	te = pd.read_csv('creditcard1_test.csv')
	X_test = te[te.columns[:-1].tolist()]
	y_test = te[te.columns[-1]]

if (rank ==1): 
	df = pd.read_csv('credit2_for_update.csv')
	X_train = df[df.columns[:-1].tolist()]
	y_train = df[df.columns[-1]]

# feed data into model and set the parameter of fl_split accoring to the size of aggregated data.
# if you do not set fl_split, then xgboost model will train as original mode.
# for the cluster number = 405, when min_child_weight = 0.2, it can achieve best performance.
# for original data, when min_child_weight = 0.01, it can achieve best performance.
FML = XGBClassifier(fl_split=405,tree_method='approx',updater='grow_histmaker,prune',learning_rate=0.1, 
					n_estimators=1000, max_depth=4, min_child_weight=0.2,gamma=0.03,subsample=0.6,nthread=4,scale_pos_weight=1,seed=27)

FML.fit(X_train, y_train)

# each node save the model.
if rank == 0 :
	y_pred = FML.predict(X_test)
	y_pred_proba = FML.predict_proba(X_test)[:, 1]
	print("Score: ", FML.score(X_test, y_test))
	print("F1 score is: {}".format(f1_score(y_test, y_pred)))
	print("AUC Score is: {}".format(roc_auc_score(y_test, y_pred_proba)))
	FML.save_model('FML-node0.model')

if rank == 1 :
	FML.save_model('FML-node1.model')


xgb.rabit.finalize()

