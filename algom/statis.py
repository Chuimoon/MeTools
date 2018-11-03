# coding : utf-8
'''
统计算法模块
'''
import numpy as np

def array_area_mean(arr,fill_value=None):
    '''
    该函数会将一个包含时间维的三维数组通过空间平均计算，降维成一个一维时间序列。

    输入参数
    -------
    arr : 三维数组  `numpy.ndarray`
        该数组为待处理的原始三维数组，其第一维须为时间维，类型须为ndarray。

    fill_value : 浮点数 `fload`
        数据缺省值，若输入数组中存在缺省值，则将该参数须设置为相应的缺省值

    返回值
    -----
    一维数组  `numpy.ndarray`
        该数组为经过计算后得到的一维区域平均数组

    示例
    ---
    >>> import numpy as np
    >>> import algom.statis as statis

    # 该语句可以直接获得样例数据
    >>> from testlib.sample_data import array_sample_path

    # 加载样例数据
    >>> sample = np.load(array_sample_path())

    # 查看样例数据的维度信息，其维度信息为 60 x 180 x 360，
    # 说明：该样例维度组成为(time,lat,lon)
    >>> sample.shape
    (60, 180, 360)

    # 查看样例数据，发现样例数据中存在大量以-9999作为值的缺省
    >>> sample[0]
    array([[-9999., -9999., -9999., ..., -9999., -9999., -9999.],
           [-9999., -9999., -9999., ..., -9999., -9999., -9999.],
           [-9999., -9999., -9999., ..., -9999., -9999., -9999.],
           ...,
           [   89.,    79.,   110., ...,    99.,    73.,   123.],
           [  103.,    89.,   117., ...,   107.,    88.,   124.],
           [  146.,    96.,    95., ...,    75.,   103.,   158.]])

    # 对样例数据进行区域平均计算，设置缺省值为-9999
    >>> area_mean = statis.array_area_mean(sample,fill_value=-9999)

    # 查看处理结果的维度信息，
    # 可以看到，经过处理后的数据已经降到一维，即只剩时间维
    >>> area_mean.shape
    (60,)

    # 查看处理后的数据值
    # 该样例数据的物理意义是液态云水路径（LWP）
    >>> area_mean
    array([ 144.8618261 ,  167.89783351,  222.64986684,  167.88972862,
            152.66417121,  148.35237346,  147.92176898,  159.62418477,
            190.85580189,  188.49299179,  163.08826766,  142.35336827,
            152.10424399,  175.15126043,  204.97411914,  190.68251399,
            148.41361916,  144.07200495,  143.36070443,  153.28463659,
            188.25254651,  184.63940359,  152.52124035,  141.72818126,
            144.98190537,  176.8913243 ,  212.27151038,  169.37740344,
            150.48879837,  145.45350781,  143.76670305,  159.03924976,
            188.65932368,  181.84297621,  155.42023686,  138.61373365,
            145.44528896,  168.6988222 ,  191.10826741,  161.59488741,
            151.86737998,  146.01663097,  143.88816537,  159.59065392,
            190.49645556,  188.33665917,  157.72631155,  137.73068253,
            147.48557969,  168.20355919,  209.6018058 ,  167.15713193,
            149.14730259,  140.69769473,  143.19762286,  156.5183357 ,
            183.84408836,  191.18815922,  154.25274534,  137.73323471])

    '''
    if fill_value:
        arr = np.ma.masked_equal(arr,fill_value)

    return np.array(np.mean(np.mean(arr,axis=1),axis=1))


def array_annual_mean(arr,fill_value=None):
    '''
    该函数用于计算三维数组的年平均值

    输入参数
    --------
    arr : 三维数组  `numpy.ndarray`
        该数组为待处理的原始三维数组，其第一维须为时间维（逐月），类型须为ndarray（numpy数组）。
        该函数默认arr数组时间维的首值为某年一月，若首值不是一月，请不要直接使用本函数，须预先将
        输入数组处理为首值为一月方可使用本函数，末值无限制，但若最后一年非完整年，则按照已有的月
        份做相应的平均，与完整年的结果会有偏差。

    返回值
    ------
    三维数组  `numpy.ndarray` | `numpy.ma.core.MaskedArray`
        该数组为经过计算后得到的年平均三维数组，其时间维缩短为逐年值，若输入参数中fill_value
        不为空，则返回的数组会根据fill_value进行掩码，得到一个numpy.ma.core.MaskedArray
        型数组，若fill_value为空，则所有数据参与计算，得到一个numpy.ndarray型数组

    示例
    ----
    >>> import numpy as np
    >>> import algom.statis as statis

    # 该语句可以直接获得样例数据
    >>> from testlib.sample_data import array_sample_path

    # 加载样例数据
    >>> sample = np.load(array_sample_path())

    # 查看样例数据的维度信息，其维度信息为 60 x 180 x 360，
    # 说明：该样例维度组成为(time,lat,lon)
    >>> sample.shape
    (60, 180, 360)

    # 查看样例数据，发现样例数据中存在大量以-9999作为值的缺省
    >>> sample[0]
    array([[-9999., -9999., -9999., ..., -9999., -9999., -9999.],
           [-9999., -9999., -9999., ..., -9999., -9999., -9999.],
           [-9999., -9999., -9999., ..., -9999., -9999., -9999.],
           ...,
           [   89.,    79.,   110., ...,    99.,    73.,   123.],
           [  103.,    89.,   117., ...,   107.,    88.,   124.],
           [  146.,    96.,    95., ...,    75.,   103.,   158.]])

    # 对样例数据进行年平均计算，设置缺省值为-9999
    >>> annual_mean = statis.array_annual_mean(sample,fill_value=-9999)

    # 年平均数组的时间维降为5，即5年（60/12=5)
    >>> annual_mean.shape
    (5, 180, 360)

    # 若fill_value存在，则返回的是一个MaskedArray
    >>> annual_mean[0]
    masked_array(data =
     [[-- -- -- ..., -- -- --]
     [-- -- -- ..., -- -- --]
     [-- -- -- ..., -- -- --]
     ...,
     [89.0 79.0 110.0 ..., 99.0 73.0 123.0]
     [103.0 89.0 117.0 ..., 107.0 88.0 124.0]
     [146.0 96.0 95.0 ..., 75.0 103.0 158.0]],
                 mask =
     [[ True  True  True ...,  True  True  True]
     [ True  True  True ...,  True  True  True]
     [ True  True  True ...,  True  True  True]
     ...,
     [False False False ..., False False False]
     [False False False ..., False False False]
     [False False False ..., False False False]],
           fill_value = -9999.0)
    '''
    lenth = int(arr.shape[0] / 12)
    ann_arr = np.full((lenth,arr.shape[1],arr.shape[2]),-9999.)
    if fill_value:
        arr = np.ma.masked_equal(arr,fill_value)
        for n in range(lenth):
            ann_arr[n] = np.ma.mean(arr[12*n:12*n+12],axis=0)
        ann_arr = np.ma.masked_equal(arr,fill_value)
    else:
        for n in range(lenth):
            ann_arr[n] = np.mean(arr[12*n:12*n+12],axis=0)

    return ann_arr


def array_smooth(lon,lat,arr,zoom=3):
    '''
    该函数用于对空间二维数组进行插值平滑处理

    输入参数
    -------
    lon : 一维数组  `numpy.ndarray` | `list`
        经度数组

    lat : 一维数组  `numpy.ndarray` | `list`
        纬度数组

    arr : 二维数组  `numpy.ndarray`
        二维空间数组

    zoom : 整数  `int`
        插值倍数

    返回值
    -----
    (slon,slat,sarr) : (一维数组，一维数组，二维数组) `tumple`
        平滑插值处理后得到的(经度数组，纬度数组，二维空间数组）
        
    '''
    sarr = scipy.ndimage.zoom(arr,zoom)
    slat = np.linspace(lat[0],lat[-1],sarr.shape[0])
    slon = np.linspace(lon[0],lon[-1],sarr.shape[1])

    return (slon,slat,sarr)


def sequence_correlation(x,y):
    '''
    该函数用于计算两个相同长度一维数组的相关系数

    输入参数
    -------
    x : 一维数组 `numpy.ndarray` | `list`
        第一个一维数组

    y : 一维数组 `numpy.ndarray` | `list`
        第二个一维数组

    返回值
    -----
    浮点数 `float`
        返回两个输入数组的线性相关系数，正常返回结果在-1~1之间，若数据异常则返回值为-9999。

    示例
    ---
    # 当两个数组不存在异常情况时
    >>> sequence_correlation([1,2,3],[1,2,3])
    1.0

    # 当两个数组至少有一个异常时,返回-9999
    >>> sequence_correlation([1,2,3],[2,2,2])
    -9999.0
    '''
    xshape = np.array(x).shape
    yshape = np.array(y).shape
    if len(xshape) != 1:
        raise ValueError('x should be a one-dimension array.')
    elif len(yshape) != 1:
        raise ValueError('y should be a one-dimension array.')
    elif xshape == yshape:
        try:
            int(np.corrcoef(x,y)[0,1])  #判断所得结果是否可数字化，若结果为nan则无法数字化将按异常处理
        except ValueError:              #若得到nan则以-9999作为返回值
            return -9999.0
        else:
            return np.corrcoef(x,y)[0,1]
    else:
        raise ValueError('x and y\'s length should be the same.')
