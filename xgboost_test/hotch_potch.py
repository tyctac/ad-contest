# -*- coding: utf-8 -*-
"""
baseline 2: ad.csv (creativeID/adID/camgaignID/advertiserID/appID/appPlatform) + lr
"""

import zipfile
import pandas as pd
import numpy as np
from scipy import sparse
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from utils import config
import datetime
import xgboost as xgb

begintime = datetime.datetime.now()

def get_data():
    # load data
    data_root = config.get_home_dir() + 'raw_data/pre'
    dfTrain = pd.read_csv("%s/train.csv"%data_root)
    dfTest = pd.read_csv("%s/test.csv"%data_root)
    dfAd = pd.read_csv("%s/ad.csv"%data_root)
    df_app_cat = pd.read_csv('%s/app_categories.csv'%data_root)

    dfuser = pd.read_csv('%s/user.csv'%data_root)
    dfuser_app_actions = pd.read_csv('%s/user_app_actions.csv'%data_root)
    dfuser_installed_apps = pd.read_csv('%s/user_installedapps.csv'%data_root)
    dfposition = pd.read_csv('%s/position.csv'%data_root)
    # process data
    dfTrain = pd.merge(dfTrain, dfAd, on="creativeID")
    print dfTrain.columns
    dfTrain = pd.merge(dfTrain, df_app_cat, on = 'appID')
    dfTrain = pd.merge(dfTrain, dfuser, on = 'userID')

    dfTest = pd.merge(dfTest, dfAd, on="creativeID")
    dfTest = pd.merge(dfTest, df_app_cat, on ='appID')
    dfTest = pd.merge(dfTest, dfuser, on = 'userID')

    y_train = dfTrain["label"].values
    return dfTrain, dfTest, y_train

def show_columns(dfTrain, dfTest):
    traincol = dfTrain.columns
    for col in traincol:
        print col, ' ',
    print ''
    testcol = dfTest.columns
    for col in testcol:
        print col, ' ',
    print ''

# feature engineering/encoding
# label  conversionTime \
#  clickTime  creativeID   userID   positionID   connectionType   telecomsOperator   adID   camgaignID   advertiserID   appID   appPlatform   appCategory   age   gender   education   marriageStatus   haveBaby   hometown   residence
# label  instanceID  \
#  clickTime  creativeID   userID   positionID   connectionType   telecomsOperator   adID   camgaignID   advertiserID   appID   appPlatform   appCategory   age   gender   education   marriageStatus   haveBaby   hometown   residence

def save(dfTrain,dfTest):
    flag = False
    for i,col in enumerate(dfTrain.columns):
        if col == 'conversionTime' or col == 'label':
            continue
        x_train = dfTrain[col].values.reshape(-1,1)
        x_test = dfTest[col].values.reshape(-1,1)
        if flag == False:
            X_train, X_test = x_train, x_test
            flag = True
        else:
            X_train, X_test = np.hstack((X_train, x_train)), np.hstack((X_test, x_test))
    np.save('X_train', X_train)
    np.save('X_test', X_test)

def predict(X_train, y_train, X_test, result_path):
    param = {'bst:max_depth': 4, 'bst:eta': 1, 'objective': 'binary:logistic'}
    param['nthread'] = 6
    plst = param.items()
    plst += [('eval_metric', 'auc')]
    # plst += [('eval_metric', 'ams@0')]
    num_round = 10
    dtrain = xgb.DMatrix(X_train, label=y_train)
    bst = xgb.train(plst, dtrain, num_round)
    bst.save_model('first_model')
    dtest = xgb.DMatrix(X_test)
    ret = bst.predict(dtest)
    np.save(result_path, ret)

def submission(y_test_path):
    dfTest = config.get_origin_test()
    y_test = np.load(y_test_path)
    # submission
    df = pd.DataFrame({"instanceID": dfTest["instanceID"].values, "proba": y_test})
    df.sort_values("instanceID", inplace=True)
    df.to_csv("submission.csv", index=False)
    with zipfile.ZipFile("submission.zip", "w") as fout:
        fout.write("submission.csv", compress_type=zipfile.ZIP_DEFLATED)

def main2():
    submission(config.get_file_path('v1','xgboost') + 'y_test.npy')


def main():
    train_path = config.get_home_dir() + 'tmp/v1/_xgboost/X_train.npy'
    y_train_path = config.get_home_dir() + 'tmp/v1/_xgboost/y_train.npy'
    test_path = config.get_home_dir() + 'tmp/v1/_xgboost/X_test.npy'
    y_test_path = config.get_file_path('v1', 'xgboost') + 'y_test'
    X_train = np.load(train_path)
    y_train = np.load(y_train_path)
    X_test = np.load(test_path)
    predict(X_train, y_train, X_test, y_test_path)

if __name__ == '__main__':
    main2()
# endtime = datetime.datetime.now()
# duringtime = endtime - begintime
# print duringtime.seconds