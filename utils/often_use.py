#encoding=utf8
import pandas as pd
from datetime import datetime,timedelta
from utils import config
# ad, position, train, user, app_cat, test, ua_ac, user_apps
def get_train_date_array():
    date1 = datetime.strptime('2016-07-19', '%Y-%m-%d')
    date2 = datetime.strptime('2016-10-17', '%Y-%m-%d')
    d1 = date1
    dates = []
    while d1 <= date2:
        if d1.month == 10 and d1.day == 10:  ##　之前因为该天天气信息缺失所以忽略该天
            d1 = d1 + timedelta(days=1)
            continue
        dates.append(d1.date())
        d1 = d1 + timedelta(days=1)
    return dates

def import_origin():
    ori_path = config.get_origin_path()
    print ori_path
    ad = pd.read_csv(ori_path + 'ad.csv')
    position = pd.read_csv(ori_path + 'position.csv')
    train = pd.read_csv(ori_path + 'train.csv')
    user = pd.read_csv(ori_path + 'user.csv')
    app_cat = pd.read_csv(ori_path + 'app_categories.csv')
    test = pd.read_csv(ori_path + 'test.csv')
    ua_ac = pd.read_csv(ori_path + 'user_app_actions.csv')
    user_apps = pd.read_csv(ori_path + 'user_installedapps.csv')
    return ad, position, train, user, app_cat, test, ua_ac, user_apps

if __name__ == '__main__':
    ad, position, train, user, app_cat, test, ua_ac, user_apps = import_origin()
    print 'ok'