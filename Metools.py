# coding: utf-8
'''
Meteorology Tools

Prg created by Clarmy 2018/06/15  clarmylee92510@gmail.com  https://github.com/Clarmy
Doc Created by Clarmy 2018/06/15  clarmylee92510@gmail.com  https://github.com/Clarmy

'''
import numpy as np
import netCDF4 as nc
import mpl_toolkits.basemap as bm
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.patches import Polygon
import shapefile
from matplotlib.path import Path
from matplotlib.patches import PathPatch
import pandas as pd
import scipy.ndimage

months = ['Jan.','Feb.','Mar.','Apr.','May','Jun.','Jul.','Aug.','Sep.','Oct.','Nov.','Dec.']

def AdjustColormap(cmap, minval=0.0, maxval=1.0, n=100):
    u'''
    This function can intercept an existing colormap and return a new colormap objection. 

    Parameters
    ----------
    cmap : a cmap objection
        It is the existing colormap for intercepting.
    minval : a float digit between 0 and 1
        It is the leftmost position of new colormap to create, minval must be less than maxval.
    maxval : a float digit between 0 and 1
        It is the rightmost position of new colormap to create, maxval must be greater than minval.
    n : a integer digit
        It is the amount of interval of new colormap to create.

    Returns
    -------
    new_cmap : a cmap objection

    '''
    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap


def AreaMean(arr):
    u'''
    该函数用于将一个三维数组计算得到一个一维的区域平均数组。

    输入参数
    --------
    arr : 三维数组
        该数组为待处理的原始三维数组，其第一维须为时间维，类型须为ndarray（numpy数组）。

    返回值
    -------
    result : 一维数组
        该数组为经过计算后得到的一维区域平均数组

    '''
    return np.array(np.mean(np.mean(arr,axis=1),axis=1))

def AnnualMeanArray(arr):
    u'''
    该函数用于计算三维数组的年平均值

    输入参数
    --------
    arr : 三维数组
        该数组为待处理的原始三维数组，其第一维须为时间维（逐月），类型须为ndarray（numpy数组）。
        该函数默认arr数组时间维的首值为某年一月，若首值不是一月，请不要直接使用本函数，须预先将
        输入数组处理为首值为一月方可使用本函数，末值无限制，但若最后一年非完整年，则按照已有的月
        份做相应的平均，与完整年的结果会有偏差。

    返回值
    ------
    ann_arr : 三维数组
        该数组为经过计算后得到的年平均三维数组，其时间维缩短为逐年值，类型仍为ndarray（numpy数组）

    示例
    ----
    示例数据为16年192个月的温度数据（temp），其维度分布为 (月份:192, 纬度:180, 经度:360)：
    In [1] : temp.shape
    Out [1] : (192, 180, 360)
    
    使用本函数计算得到年平均数组
    In [2] : annual_mean_array = AnnualMeanArray(temp)

    年平均数组的时间维降为16，即16年（192/12=16)
    In [3] : annual_mean_array.shape
    Out [3] : (16, 180, 360)
    '''
    lenth = arr.shape[0] / 12
    ann_arr = np.full((lenth,arr.shape[1],arr.shape[2]),-9999.)
    for n in xrange(lenth):
        ann_arr[n] = np.mean(arr[12*n:12*n+12],axis=0)
        
    return ann_arr  


def Corref(x,y):
    '''
    This function will return a correlation coefficient of 2 one-dimension series.

    Parameters
    ----------
    x : a one-dimension list or ndarray
        The first series.
    y : a one-dimension list or ndarray
        The second series.

    Returns
    -------
    coef : a float number
        The linear correlation coefficient

    Log
    ---
    Prg Created by Clarmy 2018/06/15  clarmylee92510@gmail.com  https://github.com/Clarmy
    Doc Created by Clarmy 2018/06/15  clarmylee92510@gmail.com  https://github.com/Clarmy
    '''
    try:
        int(np.corrcoef(x,y)[0,1])
    except ValueError:
        return -9999.0
    else:
        return np.corrcoef(x,y)[0,1]


def ClipBoundary(ctrf,ax,region='China'):
    '''
    This function will clip contourf obj inside a boundary that from a shapefile.
    '''

    existing_regions = os.listdir('./shpfiles/')
    if region in existing_regions:
        shpfile = './shpfiles/'+region+'/boundary'
    else:
        print 'ERROR: We don\'t have %s\'s shapefile.' % region
        return False

    if region == 'TibetPlateau':
        sf = shapefile.Reader(shpfile)
        for shape_rec in sf.shapeRecords():
            vertices = []
            codes = []
            pts = shape_rec.shape.points
            prt = list(shape_rec.shape.parts) + [len(pts)]
            for i in range(len(prt) - 1):
                for j in range(prt[i], prt[i+1]):
                    vertices.append((pts[j][0], pts[j][1]))
                codes += [Path.MOVETO]
                codes += [Path.LINETO] * (prt[i+1] - prt[i] -2)
                codes += [Path.CLOSEPOLY]
            clip = Path(vertices, codes)
            clip = PathPatch(clip, transform=ax.transData)
        for contour in ctrf.collections:
            contour.set_clip_path(clip)
    elif region == 'China':
        sf = shapefile.Reader(shpfile)
        for shape_rec in sf.shapeRecords():
            if shape_rec.record[3] == region:  ####这里需要找到和region匹配的唯一标识符，record[]中必有一项是对应的。
                vertices = []
                codes = []
                pts = shape_rec.shape.points
                prt = list(shape_rec.shape.parts) + [len(pts)]
                for i in range(len(prt) - 1):
                    for j in range(prt[i], prt[i+1]):
                        vertices.append((pts[j][0], pts[j][1]))
                    codes += [Path.MOVETO]
                    codes += [Path.LINETO] * (prt[i+1] - prt[i] -2)
                    codes += [Path.CLOSEPOLY]
                clip = Path(vertices, codes)
                clip = PathPatch(clip, transform=ax.transData)
        for contour in ctrf.collections:
            contour.set_clip_path(clip)

    return True

def ClipChina(ctrf,ax):
    region='China'
    shpfile = 'D:/Dropbox/Research/CloudAboveTibet/Basemaps/China/country1'
    sf = shapefile.Reader(shpfile)
    for shape_rec in sf.shapeRecords():
        if shape_rec.record[3] == region:  ####这里需要找到和region匹配的唯一标识符，record[]中必有一项是对应的。
            vertices = []
            codes = []
            pts = shape_rec.shape.points
            prt = list(shape_rec.shape.parts) + [len(pts)]
            for i in range(len(prt) - 1):
                for j in range(prt[i], prt[i+1]):
                    vertices.append((pts[j][0], pts[j][1]))
                codes += [Path.MOVETO]
                codes += [Path.LINETO] * (prt[i+1] - prt[i] -2)
                codes += [Path.CLOSEPOLY]
            clip = Path(vertices, codes)
            clip = PathPatch(clip, transform=ax.transData)
    for contour in ctrf.collections:
        contour.set_clip_path(clip)

    return clip

def Datasetize(value,region,valuename,lon,lat):
    Series = AreaMean(SelectRegion(value,region,lon,lat))

    year = []
    season = []
    month = []
    for yyyy in xrange(2001,2017):
        for mm in xrange(1,13):
            year.append(yyyy)
            month.append(mm)
            if mm in [3,4,5]:
                season.append('spring')
            elif mm in [6,7,8]:
                season.append('summer')
            elif mm in [9,10,11]:
                season.append('autumn')
            else:
                season.append('winter')

    season[0]=season[1]=None

    syear = year[:]

    syear.insert(0,None)
    syear.insert(0,None)
    syear.pop()
    syear.pop()

    dataset = pd.DataFrame({'year':year,'month':month,'season':season,'syear':syear,valuename:Series})
    dataset = dataset[['year','month','season','syear',valuename]]

    return dataset


def DrawBasemap(xticksize=15,yticksize=15,resolution='c'):
    '''
    Draw a basemap on Qinghai-Tibet Plateau and nearby area with a border of Plateau.

    DrawBasemap(fontsize=16)
    '''
    mp = bm.Basemap(llcrnrlon=71.,llcrnrlat=20.,urcrnrlon=108.,
                    urcrnrlat=45.,resolution=resolution, projection='cyl')
    mp.drawcoastlines(linewidth=0.3)
    mp.drawparallels(np.array([20,25,30,35,40,45]),
                     labels=[True,False,False,False],linewidth=0.5,
                     dashes=[1,5],fontsize=yticksize)
    mp.drawmeridians(np.array([70,75,80,85,90,95,100,105,110]),
                     labels=[False,False,False,True],linewidth=0.5,
                     dashes=[1,5],fontsize=xticksize)
    mp.readshapefile('D:/Dropbox/Research/CloudAboveTibet/Basemaps/'\
                     'TibetPlateau/DBATP_Line','Tibet',linewidth=2,color='k')

def DrawBasemap_large(xticksize=15,yticksize=15,resolution='h'):
    '''
    Draw a basemap on Qinghai-Tibet Plateau and nearby area with a border of Plateau.

    DrawBasemap(fontsize=16)
    '''
    mp = bm.Basemap(llcrnrlon=0.,llcrnrlat=0.,urcrnrlon=140.,
                    urcrnrlat=55.,resolution=resolution, projection='cyl')
    mp.drawcoastlines(linewidth=0.8)
    # mp.drawparallels(np.array([0,5,10,15,20,25,30,35,40,45]),
    #                  labels=[True,False,False,False],linewidth=0.5,
    #                  dashes=[1,5],fontsize=yticksize)
    # mp.drawmeridians(np.array([70,75,80,85,90,95,100,105,110]),
    #                  labels=[False,False,False,True],linewidth=0.5,
    #                  dashes=[1,5],fontsize=xticksize)
    mp.readshapefile('D:/Dropbox/Research/CloudAboveTibet/Basemaps/'\
                     'TibetPlateau/DBATP_Line','Tibet',linewidth=1.5,color='k')

def DrawBasemap_c(xticksize=15,yticksize=15,resolution='c'):
    '''
    Draw a basemap on Qinghai-Tibet Plateau and nearby area with a border of Plateau.

    DrawBasemap(fontsize=16)
    '''
    mp = bm.Basemap(llcrnrlon=71,llcrnrlat=20.,urcrnrlon=113.,
                    urcrnrlat=50.,resolution=resolution, projection='cyl')
    # mp.drawcoastlines(linewidth=0.3)
    mp.drawparallels(np.array([20,25,30,35,40,45]),
                     labels=[True,False,False,False],linewidth=0.5,
                     dashes=[1,5],fontsize=yticksize)
    mp.drawmeridians(np.array([70,75,80,85,90,95,100,105,110]),
                     labels=[False,False,False,True],linewidth=0.5,
                     dashes=[1,5],fontsize=xticksize)
    mp.readshapefile('D:/Dropbox/Research/CloudAboveTibet/Basemaps/China/bou2_4l',
                     'China',linewidth=1.2,color='k')
    # mp.readshapefile('D:/Dropbox/Research/CloudAboveTibet/Basemaps/'\
    #                  'TibetPlateau/DBATP_Line','Tibet',linewidth=2,color='b')


def DrawBasemap_s(xticksize=15,yticksize=15):
    mp = bm.Basemap(llcrnrlon=73,llcrnrlat=23.5,urcrnrlon=105.,
                    urcrnrlat=41,resolution='c', projection='cyl')
    # mp.drawcoastlines(linewidth=0.3)
    mp.drawparallels(np.array([20,25,30,35,40,45]),
                     labels=[True,False,False,False],linewidth=0.5,
                     dashes=[1,5],fontsize=yticksize)
    mp.drawmeridians(np.array([70,75,80,85,90,95,100,105,110]),
                     labels=[False,False,False,True],linewidth=0.5,
                     dashes=[1,5],fontsize=xticksize)
    mp.readshapefile('D:/Dropbox/Research/CloudAboveTibet/Basemaps/'\
                     'TibetPlateau/DBATP_Line','Tibet',linewidth=1,color='k')


def DrawClipedChinaCTF(lon,lat,value,levels,extend,cmap=plt.cm.RdYlBu,lineSwitch=True):
    lons,lats = np.meshgrid(lon,lat)
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(111)

    DrawBasemap_c()
    
    ctrf = plt.contourf(lons,lats,value,levels=levels,cmap=cmap,extend=extend)
    ClipChina(ctrf,ax)
    
    if lineSwitch:
        ctr = plt.contour(lons,lats,value,levels=levels,colors='k',linewidths=0.4)
        ClipChina(ctr,ax)
        
    cax = fig.add_axes([0.175,0.3,0.4,0.027])
    cb = fig.colorbar(ctrf,orientation='horizontal',cax=cax,ticks=levels[::3])
    cb.ax.tick_params(labelsize=13)

    # fig.text(0.5, 0.21, '$Longitude$', ha='center',fontsize=22)
    # fig.text(0.02, 0.5, '$Latitude$', 
    #      va='center', rotation='vertical',fontsize=22)

def DrawClipedTibetCTR(lon,lat,value,levels,cmap=plt.cm.RdYlBu,lineSwitch=True):
    lons,lats = np.meshgrid(lon,lat)
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(111)
    DrawBasemap_s()
    ctrf = plt.contourf(lons,lats,value,levels=levels,cmap=cmap)
    if lineSwitch:
        ctr = plt.contour(lons,lats,value,levels=levels,colors='k',linewidths=0.3)
        ClipTibet(ctr,ax)
    ClipTibet(ctrf,ax)
    cax = fig.add_axes([0.16, 0.32, 0.025, 0.25])
    cb = fig.colorbar(ctrf,cax)
    cb.ax.tick_params(labelsize=14)

def Draw4ContourSubplots(lon,lat,value,levels=np.arange(0,401,40),
                         colors=['#FFFFFF','#DDDDDD','#AAAAAA','#888888','#666666'],
                         extend='both',labels=['a','b','c','d'],
                         lineSwitch=True):
    '''
    Draw a 4-subplots Co
    value should be a dictionary with keys of {'spring','summer','autumn','winter'}
    '''
    lons,lats = np.meshgrid(lon,lat)
    season = ['spring','summer','autumn','winter']
    fig = plt.figure(figsize=(11,8),dpi=1000)
    for i,n in enumerate(season):
        plt.subplot(2,2,i+1)
        plt.title(labels[i])
        if i == 0:
            DrawBasemap(xticksize=0)
            # SubplotLabel(labels[i])
            ctrf = plt.contourf(lons,lats,value[n],levels=levels,colors=colors,extend=extend)
            if lineSwitch:
                ctr = plt.contour(lons,lats,value[n],
                                  levels=levels,colors='k',
                                 linewidths=0.5)
        elif i==2:
            DrawBasemap()
            # SubplotLabel(labels[i])
            ctrf = plt.contourf(lons,lats,value[n],levels=levels,colors=colors,extend=extend)
            if lineSwitch:
                ctr = plt.contour(lons,lats,value[n],
                                  levels=levels,colors='k',
                                 linewidths=0.5)
        elif i==3:
            DrawBasemap(yticksize=0)
            # SubplotLabel(labels[i])
            ctrf = plt.contourf(lons,lats,value[n],levels=levels,colors=colors,extend=extend)
            if lineSwitch:
                ctr = plt.contour(lons,lats,value[n],
                                  levels=levels,colors='k',
                                 linewidths=0.5)
        else:
            DrawBasemap(xticksize=0,yticksize=0)
            # SubplotLabel(labels[i])
            ctrf = plt.contourf(lons,lats,value[n],levels=levels,colors=colors,extend=extend)
            if lineSwitch:
                ctr = plt.contour(lons,lats,value[n],
                                  levels=levels,colors='k',
                                 linewidths=0.5)

    cax = fig.add_axes([0.2, 0.05, 0.6, 0.03])
    fig.colorbar(ctrf,cax,orientation='horizontal',ticks=levels)
    plt.xticks(fontsize=15)
    plt.subplots_adjust(wspace = 0.01,hspace = 0.06)

def Draw12ContourSubplots(lon,lat,value,levels=np.arange(0,401,40),
                         cmap=plt.cm.RdYlBu,extend='max',
                         lineSwitch=True):
    lons,lats = np.meshgrid(lon,lat)
    labels = ['Jan.','Feb.','Mar.','Apr.','May','Jun.','Jul.','Aug.','Sep.','Oct.','Nov.','Dec.']
    fig = plt.figure(figsize=(25,25))
    for i,m in enumerate(months):
        if i in [0,3,6]:
            plt.subplot(4,3,i+1)
            DrawBasemap(xticksize=0,yticksize=20)
            SubplotLabel(labels[i],fontsize=30,right=77.5)
            ctrf = plt.contourf(lons,lats,value[i],
                                levels=levels,
                                cmap=cmap,extend=extend)
            if lineSwitch:
                ctr = plt.contour(lons,lats,value[i],
                                  levels=levels,colors='k',
                                 linewidths=0.5)
            # plt.text(72.5,41.5,m,fontsize=30)
        elif i in [10,11]:
            plt.subplot(4,3,i+1)
            DrawBasemap(yticksize=0,xticksize=20)
            SubplotLabel(labels[i],fontsize=30,right=77.5)
            ctrf = plt.contourf(lons,lats,value[i],
                                levels=levels,
                                cmap=cmap,extend=extend)
            if lineSwitch:
                ctr = plt.contour(lons,lats,value[i],
                                  levels=levels,colors='k',
                                 linewidths=0.5)
            # plt.text(72.5,41.5,m,fontsize=30)
        elif i == 9:
            plt.subplot(4,3,i+1)
            DrawBasemap(xticksize=20,yticksize=20)
            SubplotLabel(labels[i],fontsize=30,right=77.5)
            ctrf = plt.contourf(lons,lats,value[i],
                                levels=levels,
                                cmap=cmap,extend=extend)
            if lineSwitch:
                ctr = plt.contour(lons,lats,value[i],
                                  levels=levels,colors='k',
                                 linewidths=0.5)
            # plt.text(72.5,41.5,m,fontsize=30)
        else:
            plt.subplot(4,3,i+1)
            DrawBasemap(xticksize=0,yticksize=0)
            SubplotLabel(labels[i],fontsize=30,right=77.5)
            ctrf = plt.contourf(lons,lats,value[i],
                                levels=levels,
                                cmap=cmap,extend=extend)
            if lineSwitch:
                ctr = plt.contour(lons,lats,value[i],
                                  levels=levels,colors='k',
                                 linewidths=0.5)
            # plt.text(72.5,41.5,m,fontsize=30)
        
    cax = fig.add_axes([0.2, 0.1, 0.6, 0.01])
    fig.colorbar(ctrf,cax,orientation='horizontal',ticks=levels)
    plt.xticks(fontsize=25)
    plt.subplots_adjust(wspace = 0.001,hspace = 0.1)


def Draw4RegressSubplots(x,data,keys,ylim,ylabel='Y-axis',xlabel='X-axis',sublabels=['(a)','(b)','(c)','(d)']):
    '''
    Draw a linear regression plot with 4 subplots.

    Draw4RegressSubplot(x,data,keys,ylim,ylabel='Y-axis',xlabel='X-axis',sublabels=['(a)','(b)','(c)','(d)'])

    x: X-axis, a 1-D array.
    data: Y-axis, a dictionary with 4 keys, each keys' value must be same as the dimension of x.
    keys: a list that be same as the data's keys, but in a order that you want to draw.
    ylim: a 1-D list or tumple with 2 integer, the first one means the y-axis' min value and the last on means
        the max value.
    ylabel: a string, the y axis' name.
    xlabel: a string, the x axis' name.
    sublabels: a list with 4 strings to label each subplot.

    '''
    fig = plt.figure(figsize=(12,7))
    for i,n in enumerate(keys):
        plt.subplot(2,2,i+1)
        plt.plot(x,data[n],'b',linewidth=1.5)
        p = LineRegression(x,data[n])
        plt.plot(x,p(x),'r--',linewidth=1.5)
        plt.scatter(x,data[n],color='b')
        plt.text(x[0],(ylim[1]-ylim[0])*0.85+ylim[0],sublabels[i],fontsize=20)
        plt.xlim(x[0]-0.5,x[-1]+0.5)
        plt.ylim(ylim[0],ylim[1])
        plt.grid()
        if i in [0,2]:
            plt.yticks(fontsize=12.5)
        else:
            plt.yticks(fontsize=0)
        if i in [2,3]:
            plt.xticks(fontsize=13)
        else:
            plt.xticks(fontsize=0)
            
        print '\nTrend of '+n+' is ',round(TrendRate(x,data[n]),4)
        print 'Coer of '+n+' is ',round(Corref(x,data[n]),4)
            
    fig.text(0.5, 0.05, xlabel, ha='center',fontsize=20)
    fig.text(0.06, 0.5, ylabel, 
             va='center', rotation='vertical',fontsize=20)
    plt.subplots_adjust(wspace = 0.02,hspace = 0.055)


def Draw12RegressSubplots(x,y,ylim,xlabel='x-axis',
                          ylabel = 'y-axis',
                          labels = ['Jan.','Feb.','Mar.','Apr.',
                                    'May','Jun.','Jul.','Aug.',
                                    'Sep.','Oct.','Nov.','Dec.']):
    fig = plt.figure(figsize=(15,10))

    trend = []
    coef = []
    for i in xrange(12):
        plt.subplot(4,3,i+1)
        plt.plot(x,y[i+1],'b',linewidth=1.5,label=labels[i])
        plt.scatter(x,y[i+1],color='b')  #在折线折点上叠加画上实心圆点
        plt.text(x[0],(ylim[1]-ylim[0])*0.85+ylim[0],labels[i],fontsize=20)

        p = LineRegression(x,y[i+1])
        plt.plot(x,p(x),'r--',linewidth=1.5)  #画趋势线

        plt.xlim(2000.5,2016.5)  #调节x轴范围
        plt.ylim(ylim[0],ylim[1])
        plt.grid()
        if i in (9,10,11):
            plt.xticks(x[1::3],fontsize=13)  #设置x轴刻度
        else:
            plt.xticks(x[1::3],fontsize=0)
        if i in (0,3,6,9):
            plt.yticks(fontsize=13)
        else:
            plt.yticks(fontsize=0)

        print '\nTrend of '+labels[i]+' is ',round(TrendRate(x,y[i+1]),4)
        print 'Coer of '+labels[i]+' is ',round(Corref(x,y[i+1]),4)

    plt.subplots_adjust(wspace = 0.02,hspace = 0.05)

    fig.text(0.5, 0.075, xlabel, ha='center',fontsize=20)
    fig.text(0.07, 0.5, ylabel, 
             va='center', rotation='vertical',fontsize=20)


def DrawColorbar(im,pad='6%',fontsize=15):
    divider = make_axes_locatable(plt.gca())
    cax = divider.append_axes('bottom', '5%', pad=pad)
    cb = plt.colorbar(im,orientation='horizontal',cax=cax)
    cb.ax.tick_params(labelsize=fontsize)

def LineRegression(x,y):
    '''
    Return a regression expression function.

    Regression(x,y,n)
    x: 1 dimension data on x axis and it's length must be same as y.
    y: 1 dimension data on y axis and it's length must be same as x.

    example:
    In[1] :a = [1,2,3,4,5]
    In[2] :b = [23.4,23.2,84.3,35.6,28.9]

    In[3] :f = LineRegression(a,b)

    In[4] :f(a)
    Out[2]:array([ 34.4 ,  36.74,  39.08,  41.42,  43.76])

    '''
    z = np.polyfit(x,y,1)
    p = np.poly1d(z)
    return p


def Monthlize(data,dicSwitch=True):
    months = ['Jan.','Feb.','Mar.','Apr.','May','Jun.','Jul.','Aug.','Sep.','Oct.','Nov.','Dec.']
    Series = []
    for n in xrange(12):
        datam = []
        for i in xrange(n,data.shape[0],12):
            try:
                datam.append(data[i])
            except IndexError:
                pass

        Series.append(np.ma.mean(datam,axis=0))

    Series = np.ma.MaskedArray(Series,fill_value=-9999.)
    Dic = {}
    for i,m in enumerate(months):
        Dic[m] = Series[i]

    if dicSwitch:
        return Dic
    else:
        return Series

def Seasonalize(data):
    seasons = ['spring','summer','autumn','winter']
    Monthly = Monthlize(data,dicSwitch=False)
    Seasonal = {}
    for i,n in enumerate(seasons):
        Seasonal[n] = np.mean(Monthly[i*3:i*3+3],axis=0)
    
    return Seasonal


def SelectRegion(arr,pdDF,lon,lat):
    '''
    return a array within your selected area. Outside your selected 
    point will be filled by -9999.

    SelectRegion(arr,pdDF,lon,lat)
    arr: The original 3-D array.
    pdDF: A 2 columns pandas DataFrame that labeled by 'X' and 'Y'.
    lon: The longitude of original array.
    lat: The latitude of original array.

    Example:

    fileObj = nc.Dataset('../Data/CWP/cwp0116.nc')
    tibetRegion = pd.read_csv('../Data/Region/TibetGrids.csv')

    lat = fileObj.variables['lat'][:]
    lon = fileObj.variables['lon'][:]
    lcwp = fileObj.variables['LCWP'][:]
    
    rglcwp = SelectRegion(lcwp,tibetRegion,lon,lat)

    '''
    newarr = np.full_like(arr,-9999.)
    selcoord = zip(pdDF['Y'],pdDF['X'])
    if len(arr.shape) == 3:
        for t in xrange(arr.shape[0]):
            for iy,r in enumerate(lat):
                for ix,c in enumerate(lon):
                    for crd in selcoord:
                        if crd[0] == r and crd[1] == c:
                            newarr[t,iy,ix] = arr[t,iy,ix]
    else:
        for iy,r in enumerate(lat):
                for ix,c in enumerate(lon):
                    for crd in selcoord:
                        if crd[0] == r and crd[1] == c:
                            newarr[iy,ix] = arr[iy,ix]

    return np.ma.masked_equal(newarr,-9999.)

def Smooth(data,lon,lat,zoom=3):
    
    data = scipy.ndimage.zoom(data,zoom)
    slat = np.linspace(lat[0],lat[-1],data.shape[0])
    slon = np.linspace(lon[0],lon[-1],data.shape[1])
    
    return (data,slon,slat)


def SubplotLabel(string,left=72,right=76.5,lower=40.5,upper=43.5,fontsize=20,zorder=3):
    '''
    Draw a white background of labels on a map.
    SubplotLabel(string,left=72,right=76.5,lower=40.5,upper=43.5,zorder=3)
    '''
    lower_left = (left, lower)
    lower_right= (right, lower)
    upper_left = (left, upper)
    upper_right= (right, upper)
    p = Polygon([lower_left,upper_left,upper_right,lower_right],
        facecolor='w',linewidth=0,zorder=zorder) 
    plt.gca().add_patch(p)
    plt.text(72.5,41.5,string,fontsize=fontsize)


def TrendArray(value):
    x = range(value.shape[0])
    shape = (value.shape[1],value.shape[2])
    regression = np.full(shape,-9999.)

    for iy in xrange(shape[0]):
        for ix in xrange(shape[1]):
            regression[iy,ix] = TrendRate(x,value[:,iy,ix])

    return regression

def TrendRate(x,y):
    '''
    Return a linear trend rate of y-axis data with y-axis data.
    '''
    return np.polyfit(x,y,1)[0]


def Unpack(Obj):
    offset = Obj.add_offset
    factor = Obj.scale_factor
    return offset + Obj[:] * factor

def CoefArray(value):
    x = range(value.shape[0])
    shape = (value.shape[1],value.shape[2])
    coef = np.full(shape,-9999.)

    for iy in xrange(shape[0]):
        for ix in xrange(shape[1]):
            coef[iy,ix] = Corref(x,value[:,iy,ix])

    return coef

def DrawClipScatter(ax,lon,lat,sizes=20,color='b',alpha=1):
    lat = [n for n in lat if n>20]
    lon = [n for n in lon if n<113]
    lons,lats = np.meshgrid(lon,lat)
    region='China'
    shpfile = 'D:/Dropbox/Research/CloudAboveTibet/Basemaps/China/country1'
    sf = shapefile.Reader(shpfile)
    for shape_rec in sf.shapeRecords():
        if shape_rec.record[3] == region:  ####这里需要找到和region匹配的唯一标识符，record[]中必有一项是对应的。
            vertices = []
            codes = []
            pts = shape_rec.shape.points
            prt = list(shape_rec.shape.parts) + [len(pts)]
            for i in range(len(prt) - 1):
                for j in range(prt[i], prt[i+1]):
                    vertices.append((pts[j][0], pts[j][1]))
                codes += [Path.MOVETO]
                codes += [Path.LINETO] * (prt[i+1] - prt[i] -2)
                codes += [Path.CLOSEPOLY]
            clip = Path(vertices, codes)
            clip = PathPatch(clip, transform=ax.transData)
    sct = ax.scatter(lons,lats,s=sizes,color=color,alpha=alpha)

    sct.set_clip_path(clip)
