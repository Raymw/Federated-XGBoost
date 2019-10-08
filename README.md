# Federated-XGBoost
Federated Learning on XGBoost
## 
This repository is the implementation of the paper " [The Tradeoff Between Privacy and Accuracy in Anomaly Detection Using Federated XGBoost](https://arxiv.org/abs/1907.07157) ".

## Dataset

Dataset used in this experiment is from kaggle : [Credit Card Fraud Detection](https://www.kaggle.com/mlg-ulb/creditcardfraud).

The aggregated data in the process of data aggregation has been put into the Data folder. Inside each aggregated data folder, there are three files.   `creditcard1_train` and `creditcard1_test` is used for training and in server node. `creditcard2` and  `credit2_for_update` are the dataset in local node.
 `credit2_for_update` is the data `creditcard2` after the process of federated update. The index of each aggregated data folder is the sequence we want to set in process of  finding split.

## Run

We are implmenting the horizontal federated learning scenario based on [XGBoost](https://github.com/dmlc/xgboost). Firstly, download the XGBoost package following the [XGBoost official documentation](https://xgboost.readthedocs.io/en/latest/). 

In order to satisfy the federated framework of our paper, there are two files that need to be modified. File `param.h` and `updater_histmaker.cc` have been put into folder Code. Mainly changes are to set a new parameter for our aggregated sequence in file `param.h` (Line 99 and 103) and add the corresponding way of creating split set in the paper (changes are in `updater_histmaker.cc` Line 297, 315 and 493). Remember to recompile it after adding changes.

Some other steps may also need to be done.  Comment Line 11、53 and 54 in file `submit.py` because there are errors when running code. Then, do`make` in folder Rabit.
