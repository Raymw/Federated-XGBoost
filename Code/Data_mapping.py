#Author Mengwei Yang
# this is to achieve simple data mapping

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import warnings
warnings.simplefilter("ignore")
import pandas as pd

# to set the cluster number for different feature. 
# Actually the upper bound and lower bound of each future can also be set seperatly 
# and mapped into different sequence field. But it still needs to keep the required total sequence number 

def ran(X):
    Max_F = max(X)
    Min_F = min(X)
    sum= abs(Max_F) + abs(Min_F)
    step= sum/405
    F_min.append(Min_F)
    F_step.append(step)


F_min = []
F_step = []

# in this mapping process, we did not use full dataset to do mapping. Instead, we use creditcard1_train.csv to do mapping. 
# in that case, there may have some numbers of other nodes's datasets that exceed the upper bound of our parameter fl_split. 
# We were doing this because we want to simulate that some values of instances may exceed the normal limit. 
# So in that case, with our dataset, we find the number of these instances is quite small compared with the whole dataset
# and the performance will be better if we put those instances into leftnode when finding split. 

PE = pd.read_csv('creditcard1_train.csv')

for h in range (29):
    X = PE.iloc[:,h].values.tolist()
    ran(X)

len=PE.shape[0]
for i in range(29):
    for h in range(len):
        Tem=PE.iloc[h, i]
        Tem= round(abs((Tem- F_min[i])) / F_step[i])
        PE.iloc[h, i] = Tem

PE.to_csv('creditcard1_train.csv',index=False)
