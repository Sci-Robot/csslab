# -*- coding:utf-8 -*-
'''
目的：提供一些文件操作 和 数据处理的方法

方法：
    ----------功能 ------------ 方法 ---------------
    * 读取大的数据表(csv) - read_csv
    * 获取目录下所有某类型的文件名 - get_filename
    * 读取目录下所有某类型的数据（csv,xlsx） - connect_file
    * 数据表随机长度的抽样 - random_dataframe_sample
    * 数据表根据字段过滤 - dataframe_filter
    * 数据表根据时间戳过滤 - dataframe_slice_by_timestamp
    * 计算概率密度分布 - distribution
    * 计算累计概率密度分布 - distribution_cp
    * 数据归一化到某个区间 - normlize

备注：
    * 2017.10.16 - dataframe_filter方法还需要修改

'''

import os
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def read_csv(readpath):
    '''
    分块读取大的csv
    :param readpath: filepath
    :return: pd.DataFrame
    '''
    print(' - - start to read - - %s'%readpath)
    reader = pd.read_csv(readpath,iterator=True)
    loop = True
    chunkSize = 100000
    chunks = []
    while loop:
        try:
            chunk = reader.get_chunk(chunkSize)
            chunks.append(chunk)
        except StopIteration:
            loop = False
    read_d = pd.concat(chunks,ignore_index=True)
    return read_d

def get_filename(dir, filetype='.csv'):
    #返回某个目录下所有后缀为指定类型的文件名的list，不包含后缀
    filelist = []
    for filename in os.listdir(dir):
        if os.path.splitext(filename)[1] == filetype:
            filelist.append(os.path.splitext(filename)[0])
    return filelist

def connect_file(dir, connect_nums=None, filetype='.csv',header = 0):
    '''
    连接某一个目录下的所有数据，可以指定文件数量
    :param dir: the directory of files
    :param connect_nums:  需要连接的文件序号，如果为空，则默认为全部文件,例如第1，2，5个。[0,1,4]
    :param filetype: the type of files you want to get all data from
    :return: Dataframe of all data
    '''
    filelist = get_filename(dir, filetype=filetype)
    all_data = []

    if connect_nums is None:
        connect_nums = range(len(filelist))
    if dir[-1] != '\\':
        dir = dir + '\\'
    #序号超出文件数量，还有其他很多情况，学学try的用法吧，2017.6.12
    elif max(connect_nums) > len(filelist) - 1:
        print('- - Mistake - - Connect num is wrong')
        return None

    if len(filelist) != 0:
        for file_id in connect_nums:
            readpath = dir + filelist[file_id] + filetype
            #注意有的文件的encoding类型
            print(' -- connecting file : %s '%readpath)
            each_data = pd.read_csv(readpath, header=header)
            all_data.append(each_data)
    else:
        print('- - - - - No file found - - - - - ')
        return None
    return pd.concat(all_data,ignore_index=True)

def random_dataframe_sample(df, num_sample):
    '''
    :param df: DataFrame，数据表
    :param num_sample: 随机抽样数量
    :return: DataFrame，抽取的样本数据表
    '''
    inds = list(df.index)
    if num_sample <= len(df):
        ind_sample = random.sample(inds, num_sample)
        df_sample = df.ix[ind_sample, :]
    else:
        df_sample = df
    return df_sample

def dataframe_filter(data, attr, lower = None, upper = None):
    '''
    :param data:  pandas.Dataframe obeject.
    :param attr:  the attribution that you wanna choose for filt
    :param lower: the lower limit of data
    :param upper: the upper limit of data
    :return: pandas.Dataframe after filt
    '''
    origin_length = len(data)
    if lower is not None:
        data = data[data[attr] >= lower]
    if upper is not None :
        data = data[data[attr] <= upper]

    filt_length = len(data)
    left_rate = round(filt_length/origin_length,3)

    print('- - Filter: ',lower, ' ---- ', upper)
    print('- - orgin data: %d  - - after filtering: %d  - - %.3f left'
              %(origin_length,filt_length,left_rate))
    return data

def dataframe_slice_by_timestamp(df,filter_dict):
    '''
    根据时间戳截取DataFrame数据表
    :param df: DataFrame
    :param filter_dict: dict, e.g. {'Timestamp':['2008-10-1 10;00:00','2008-10-1 12:00:00'],}
    :return: DataFrame slice
    '''
    ind_all = []
    for attr,attr_vlaue in filter_dict.items():
        timedata = df[[attr]].copy()
        timedata['IndexValue'] = list(timedata.index.values)
        timedata.index = list(timedata[attr].values)
        timedata = timedata[attr_vlaue[0]:attr_vlaue[1]]
        ind_got = timedata['IndexValue']
        ind_all.append(list(ind_got.values))
    #get intersection of those index
    if len(ind_all) > 1:
        set_ind1 = set(ind_all[0])
        ind_return = list(set_ind1.intersection(*ind_all[1:]))
    else:
        ind_return = ind_all[0]
    df = df.ix[ind_return,:]
    return df

def distribution(data):
    '''
    计算数据的概率密度分布
    :param data: list 或者 pandas.Series.
    :return: pandas.Series
    '''
    if not isinstance(data,pd.Series):
        data = pd.Series(data)
    data_count = data.value_counts().sort_index()
    data_p = data_count/data_count.sum()
    return data_p

def distribution_cp(data):
    '''
    计算累计概率密度分布
    :param data: list 或者 Series
    :return: pandas's Series
    '''
    if not isinstance(data,pd.Series):
        data = pd.Series(data)
    data_count = data.value_counts().sort_index()
    data_fre = data_count/data_count.sum()
    origin_index = data_fre.index.values
    data_fre.index = range(len(data_fre))
    data_cp = data_fre.copy()
    #分布函数，X < x
    for i in range(len(data_fre)):
        data_cp[i] = sum(data_fre[:i])
    data_cp.index = origin_index
    return data_cp

def plot_distribution(data, subplot=2, data_norm=False, cmp=False, grid=True):
    '''
    :param data: Series数据
    :param subplot: 绘制原始的，log 和 log-log
    :param data_norm: 数据是否归一化，例如normlized degree
    :param cmp: 是否绘制累计密度概率
    :param grid: 网格线是否显示
    :return: None
    '''

    if data_norm:
        data_normed = normlize(data.values, 0, 1)
        name = 'Normlized' + str(data.name)
        data = pd.Series(data_normed, name=name)

    ylabel = 'Probability'

    if cmp:
        data = distribution_cp(data)
        ylabel = 'Cumulative ' + ylabel
    else:
        data = distribution(data)

    fg = plt.figure()
    ax1 = []
    for i in range(subplot):
        ax1.append(fg.add_subplot(1, subplot, i + 1))

    data.plot(ax=ax1[0], style='*-')
    ax1[0].set_title('Distribution')

    if subplot >= 2:
        data.plot(ax=ax1[1], style='*', logy=True, logx=True)
        ax1[1].set_title('log-log')
        # ax1[1].set_xlim([0, 50])

    if subplot >= 3:
        data.plot(ax=ax1[2], style='*-', logy=True)
        ax1[2].set_title('semi-log')

    for i in range(subplot):
        ax1[i].set_ylabel(ylabel)
        ax1[i].set_xlabel(data.name)
        ax1[i].set_xlim([0, max(data.index) * 1.1])
        ax1[i].grid(grid, alpha=0.8)


def normlize(data,lower=0,upper=1):
    '''
    将数据规范化到某个区间
    :param data: 可以是list，array, ndarray等
    :param lower: 规范化的下界
    :param upper: 规范化的上界
    :return: 规范化的数据
    '''
    xmax = np.max(data)
    xmin = np.min(data)
    data_new = (upper - lower) * (data - xmin) / (xmax - xmin) + lower
    return data_new
