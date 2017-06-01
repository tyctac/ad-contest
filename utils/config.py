# encoding=utf8
import yaml
import pandas as pd
import sys,os


def get_home_dir():  # 注意  在这种情况下 字符串'tianchi'只能出现在项目根目录中,否则会出错
    oristr = os.getcwd()
    return oristr.split('ad-contest')[0] + 'ad-contest/'


def get_file_path(version1='v1', model='xgboost', version2='v1'):
    version1 += '/'
    model = '_' + model
    model += '/'
    version2 += '/'
    return get_home_dir() + 'tmp/' + version1 + model + version2


def get_origin_test(): # 获得原始的test DataFrame
    path = get_home_dir() + '/raw_data/pre/test.csv'
    return pd.read_csv(path)

def get_title_weight():
    hir = get_home_dir()
    f = open(hir + '/utils/config.yaml')
    x = yaml.load(f)
    f.close()
    return x['TITLE_WEIGHT']


def main():
    x =  get_home_dir()
    print x


if __name__ == '__main__':
    main()