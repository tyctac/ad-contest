#encoding=utf8
import yaml
import sys,os

def get_home_dir(): ## 注意  在这种情况下 字符串'tianchi'只能出现在项目根目录中,否则会出错
    oristr = os.getcwd()
    return oristr.split('ad-contest')[0] + 'ad-contest/'

def get_title_weight():
    hir = get_home_dir()
    f = open(hir + '/utils/config.yaml')
    x = yaml.load(f)
    f.close()
    return x['TITLE_WEIGHT']

def get_weather_weight():
    f = open(get_home_dir() + '/utils/config.yaml')
    x = yaml.load(f)
    f.close()
    return x['weather_weight']

def main():
    x =  get_home_dir()
    print x


if __name__ == '__main__':
    main()