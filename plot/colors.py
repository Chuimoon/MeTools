# coding : utf-8
'''
绘图调色模块
'''
import numpy as np
import matplotlib.colors as colors


def cut_cmap(cmap, minval=0.0, maxval=1.0, n=100):
    u'''
    该函数用于截取已有色标(colormap)的片段

    输入参数
    -------
    cmap : colormap对象
        该参数传入待截取的colormap对象

    minval : 浮点数 | 默认值为0
        截取片段的左边界，该数值须在0-1之间，且小于maxval

    maxval : 浮点数 | 默认值为1
        截取片段的右边界，该数值须在0-1之间，且大于minval

    n : 整数 | 默认值为100
        新色标的阶数

    返回值
    -----
    new_cmap : colormap对象
        截取后得到的colormap对象

    示例
    ----
    In[1] : old_cmap = plt.cm.Blues
    In[2] : new_cmap = cut_cmap(old_cmap,0.2,0.8)  #截取old_cmap中20%到80%之间的部分
    '''
    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))

    return new_cmap
