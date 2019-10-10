# Federated-XGBoost
Federated Learning on XGBoost
# 
This repository is the implementation of the paper " [The Tradeoff Between Privacy and Accuracy in Anomaly Detection Using Federated XGBoost](https://arxiv.org/abs/1907.07157) ".

## Dataset

Dataset used in this experiment is from kaggle : [Credit Card Fraud Detection](https://www.kaggle.com/mlg-ulb/creditcardfraud).

The aggregated data in the process of data aggregation has been put into the Data folder. Inside each aggregated data folder, there are three files.   `creditcard1_train` and `creditcard1_test` is used for training and in server node. `creditcard2` and  `credit2_for_update` are the dataset in local node.
 `credit2_for_update` is the data `creditcard2` after the process of federated update. The index of each aggregated data folder is the sequence we want to set in process of  finding split.

## Run

We are implmenting the horizontal federated learning scenario based on [XGBoost](https://github.com/dmlc/xgboost). 

Firstly, download the XGBoost package following the [XGBoost official documentation](https://xgboost.readthedocs.io/en/latest/).  In order to achieve the federated framework of our paper, there are two files that need to be modified. File `param.h` and `updater_histmaker.cc` have been put into folder Code. Exact location is in `xgboost/python-package/xgboost/src/tree`.

Main changes are to set a new parameter for our aggregated sequence in file `param.h` (Line 99 and 103) and add the corresponding way of creating split set in the paper (changes are in `updater_histmaker.cc` Line 297, 315 and 493). Remember to recompile it after adding changes.

Some other steps may also need to be done before running.  Comment Line 11、53 and 54 in file `submit.py` because there are some errors when running code. Also, we need XGBoost rabit to achieve communications between nodes. So, `cd  /xgboost/rabit` and do `make`.

We add parameter `fl_split` in federated XGBoost, which is used to set the cluster number for training. For example, in our file `data_map405`, we map the original instances values into new sequence with 405 clusters. So you can set up that parameter for our aggregated dataset. Like 

> XGBClassifier( fl_split=405, tree_method='approx'.....)

Federated model is produced by 
> /xgboost/dmlc-core/tracker/dmlc-submit --cluster mpi --num-workers 2 python main.py

Since the dataset is small, we run it locally and set only two nodes (num-workers =2) to simulate the experiments. You can change nodes parameters according to your enviroment.
