# coding : utf-8
'''
样例数据测试模块
'''
import os

def array_sample_path():
    '''获取array_sample的路径'''
    pwd = os.path.dirname(os.path.abspath(__file__))
    metools_path = '/'.join(pwd.split('/')[:-1])
    tailpath = '/data/sample/array_sample.npy'

    return metools_path + tailpath
