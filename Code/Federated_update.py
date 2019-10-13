
# Author Mengwei Yang

import pandas as pd
import xgboost as xgb
from sklearn.metrics import f1_score
import csv



def create_udate_csv():

    path = "credit2_for_update.csv"
    with open(path,'w') as f:
        csv_write = csv.writer(f)
        csv_head = ['V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10',
       'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20',
       'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'Amount',
       'Class']
        csv_write.writerow(csv_head)

# calculate the fl-score for creditcard2.csv
def calculate_f1(y,t):

    y_result = [1. if i > 0.5 else 0. for i in y]
    print('f1score for creditcard2 =',f1_score(t,y_result))
    return (y_result)

# put wrongly classified instances in creditcard2 into credit2_for_update.csv
# this filter process may need to do more than one round, however be careful not to be overfitting.
def compare(list1,list2,csvFile,te):

    writer = csv.writer(csvFile)
    if len(list1) == len(list2):
        for i in range(0, len(list1)):
            if list1[i] == list2[i]:
                pass
            else:
                writer.writerow(te.iloc[i, :])


te = pd.read_csv('creditcard2.csv')
X_test = te[te.columns[:-1].tolist()]
y_test = te[te.columns[-1]]
X_test = xgb.DMatrix(X_test)

FML = xgb.Booster(model_file='FML-node1.model')

y_pred = FML.predict(X_test)
f1_pred= calculate_f1(y_pred,y_test)


create_udate_csv()
csvFile = open("credit2_for_update.csv", "a")
compare(f1_pred,y_test,csvFile,te)




