import numpy as np
from sklearn.feature_extraction import DictVectorizer

from UGPFM_FAST import pylibfm
#from UGPFM_FAST import test

import sys
import pandas as pd
from sklearn.model_selection import train_test_split
import argparse

import time


def parse_args():
	parser = argparse.ArgumentParser(description="Run UGPFM.")
	parser.add_argument('--lr', type=float, default=0.001)
	parser.add_argument('--iter', type=int, default=100)
	parser.add_argument('--fact', type=int, default=10)
	return parser.parse_args()

args=parse_args()

#data=pd.read_csv('./level/data_difficulty_user.csv')
data=pd.read_csv('./coursera/review_complete_data_final.csv')



#X_train, X_test, y_train, y_test = train_test_split(data[['CourseId','user_id','Level_difficulty']], data['rating'], test_size = 0.12, shuffle=True)  
X_train, X_test, y_train, y_test = train_test_split(data[['CourseId','Cogn','sentiment','confusion']], data['Score'], test_size = 0.12, shuffle=True, stratify=data['Score'])  

dataFM=[]
y=[]

for index, row in X_train.iterrows():

	#dataFM.append({ "CourseId": str(row['CourseId']), "user_id": str(row['user_id']), "Level_difficulty": str(row['Level_difficulty'])})
	dataFM.append({ "CourseId": str(row['CourseId']),"Cogn": str(row['Cogn']),"sentiment": str(row['sentiment']),"confusion": str(row['confusion'])})
	y.append(float(y_train[index]))

dataFM_test=[]
y_t=[]


for index, row in X_test.iterrows():
	

	#dataFM_test.append({ "CourseId": str(row['CourseId']), "user_id": str(row['user_id']), "Level_difficulty": str(row['Level_difficulty'])})
	dataFM_test.append({ "CourseId": str(row['CourseId']),"Cogn": str(row['Cogn']),"sentiment": str(row['sentiment']),"confusion": str(row['confusion'])})
	y_t.append(float(y_test[index]))


v = DictVectorizer()
X_train = v.fit_transform(dataFM)
X_teste = v.transform(dataFM_test)


start1 = time.time()



ugpfm = pylibfm.UGPFM(num_factors=args.fact, num_iter=args.iter, verbose=True, task="regression", initial_learning_rate=args.lr, learning_rate_schedule="optimal")




ugpfm.fit(X_train,y)

end1 = time.time()

# Evaluate
preds1 = ugpfm.predict(X_teste)


# Evaluate
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score


print("****************UGPFM***********************")
print("UGPFM MSE: %.5f" % mean_squared_error(y_t,preds1))
print("UGPFM RMSE: %.5f" % mean_squared_error(y_t,preds1,squared=False))
print("UGPFM MAE: %.5f" % mean_absolute_error(y_t,preds1))
print("UGPFM R² score: %.5f" % r2_score(y_t,preds1))
T1=end1-start1
print("time of execusion of UGPFM in secondes: %d" %T1)