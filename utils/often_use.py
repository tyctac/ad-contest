#encoding=utf8
from datetime import datetime,timedelta

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
