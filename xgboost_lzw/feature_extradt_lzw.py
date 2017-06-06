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
    # dfuser_app_actions = pd.read_csv('%s/user_app_actions.csv'%data_root)
    # dfuser_installed_apps = pd.read_csv('%s/user_installedapps.csv'%data_root)
    # dfposition = pd.read_csv('%s/position.csv'%data_root)
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