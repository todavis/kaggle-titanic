# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-05-26T17:39:55.046848Z","iopub.execute_input":"2021-05-26T17:39:55.047357Z","iopub.status.idle":"2021-05-26T17:39:55.058384Z","shell.execute_reply.started":"2021-05-26T17:39:55.047247Z","shell.execute_reply":"2021-05-26T17:39:55.057313Z"}}
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-05-26T17:40:09.531361Z","iopub.execute_input":"2021-05-26T17:40:09.533419Z","iopub.status.idle":"2021-05-26T17:40:09.563806Z","shell.execute_reply.started":"2021-05-26T17:40:09.533377Z","shell.execute_reply":"2021-05-26T17:40:09.563066Z"}}
# data loader

class dataLoader:
    """
    Load data 
    """
    def __init__(self, train_filename, test_filename, normalize = ['Age', 'Fare']):
        
        self.train = pd.read_csv(train_filename)
        self.test  = pd.read_csv(test_filename)
        
        for cts_variable in normalize:
            self.normalize(cts_variable)
        
        #self.remove_na()
        #self.train_label = self.train.pop('Survived')
    
    def remove_na(self):
        """ Remove empty or invalid elements"""
        self.train.fillna(0, inplace = True)
        self.test.fillna(0, inplace = True)
    
    def normalize(self, column_name):
        """ Normalize continuous data to zero mean and standard deviation of 1"""
        
        mu = self.train[column_name].mean()
        std = self.train[column_name].std()
        
        self.train[column_name] = (self.train[column_name] - mu) / std
        self.test[column_name] = (self.test[column_name] - mu) / std
    
    def summary(self):
        """ Output summary of data and first few rows"""
        print('Training set:')
        #print(self.train.head())
        print(self.train.describe())
        
        print('Testing set:')
        #print(self.test.head())
        print(self.test.describe())
        
    
raw_data = dataLoader('/kaggle/input/titanic/train.csv', '/kaggle/input/titanic/test.csv')
raw_data.train
#raw_data.test

test_ids = raw_data.test[['PassengerId']]
#test_ids

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-05-20T00:59:37.97603Z","iopub.execute_input":"2021-05-20T00:59:37.976434Z","iopub.status.idle":"2021-05-20T00:59:38.010467Z","shell.execute_reply.started":"2021-05-20T00:59:37.976398Z","shell.execute_reply":"2021-05-20T00:59:38.009226Z"}}
# remove features that are not useful for now
raw_data.train.drop(columns = ['PassengerId', 'Ticket', 'Name'], inplace=True)
raw_data.test.drop(columns = ['PassengerId', 'Ticket', 'Name'], inplace=True)

# take the first element of the cabin feature
raw_data.train['Cabin'] = raw_data.train['Cabin'].str[0]
raw_data.test['Cabin'] = raw_data.test['Cabin'].str[0]

# encode class features using one hot encoding - try combining test to get same number of splits
raw_data.train = pd.get_dummies(raw_data.train, columns = ['Sex', 'Embarked', 'Cabin'], drop_first = True)
raw_data.test = pd.get_dummies(raw_data.test, columns = ['Sex', 'Embarked', 'Cabin'], drop_first = True)

# remove nan values from age, ...
raw_data.remove_na()
train_label = raw_data.train.pop('Survived')

# ensure train and test sets have same number of columns
missing_cols = set( raw_data.train.columns ) - set( raw_data.test.columns )
# Add a missing column in test set with default value equal to 0
for c in missing_cols:
    raw_data.test[c] = 0
# Ensure the order of column in the test set is in the same order than in train set
raw_data.test = raw_data.test[raw_data.train.columns]

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-05-20T00:59:41.228705Z","iopub.execute_input":"2021-05-20T00:59:41.229075Z","iopub.status.idle":"2021-05-20T00:59:41.323234Z","shell.execute_reply.started":"2021-05-20T00:59:41.229043Z","shell.execute_reply":"2021-05-20T00:59:41.322127Z"}}
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(raw_data.train, train_label, test_size=0.3, random_state=0)

# create model LR
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(random_state=0)
model = model.fit(X_train, y_train)
test_acc = model.score(X_test, y_test)
test_acc

# %% [code] {"execution":{"iopub.status.busy":"2021-05-20T00:59:43.904763Z","iopub.execute_input":"2021-05-20T00:59:43.905167Z","iopub.status.idle":"2021-05-20T00:59:43.941731Z","shell.execute_reply.started":"2021-05-20T00:59:43.905124Z","shell.execute_reply":"2021-05-20T00:59:43.940554Z"}}
# create svm model

from sklearn import svm
model = svm.SVC(kernel = 'rbf', degree = 2, verbose = True)
model = model.fit(X_train, y_train)
test_acc = model.score(X_test, y_test)
test_acc

# %% [code] {"execution":{"iopub.status.busy":"2021-05-20T00:59:46.161654Z","iopub.execute_input":"2021-05-20T00:59:46.162058Z","iopub.status.idle":"2021-05-20T00:59:46.350023Z","shell.execute_reply.started":"2021-05-20T00:59:46.162021Z","shell.execute_reply":"2021-05-20T00:59:46.348974Z"}}
# create RF model

from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators =80, max_depth = 9, random_state = 0)
model = model.fit(X_train, y_train)
test_acc = model.score(X_test, y_test)
test_acc

# %% [code] {"execution":{"iopub.status.busy":"2021-05-20T01:13:46.645901Z","iopub.execute_input":"2021-05-20T01:13:46.646345Z","iopub.status.idle":"2021-05-20T01:13:46.677923Z","shell.execute_reply.started":"2021-05-20T01:13:46.646278Z","shell.execute_reply":"2021-05-20T01:13:46.67689Z"}}
# save predictions from most recent model
test_pred = model.predict(raw_data.test)
#np.shape(test_pred)

submission = test_ids
submission["Survived"] = test_pred

submission.to_csv('/kaggle/working/submission.csv', index = False)
submission