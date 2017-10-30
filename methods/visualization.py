# -*- coding:utf-8 -*-
'''
目的: 提供一些可视化的方法

方法:
    * 绘制热力图
        - draw_heatmap - 绘制热图，支持多个热图绘制在1个figure上
        - heatmap_from_dataframe - 直接从dataframe绘制热图

'''

from matplotlib import cm
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def draw_heatmap(datas, mfrow, draw_axlabel=False,
                      xlabels=None, ylabels=None, xlocator=1,ylocator=1,title=list()):
    '''
    可以将多个热力图分割绘制

    :param datas: list of ndarray，如果为一张图datas设置成[data]，多张图为[data1,data2,...]
    :param mfrow: 分割图像，如(1,2)，为2个图像按1行2列的方式排列如果为1张图，设置mfrow为(1,1)
    :param draw_axlabel: 是否绘制坐标文本
    :param xlabels: x坐标的文本
    :param ylabels: y坐标的文本
    :param xlocator: x的刻度,放大，缩小刻度
    :param ylocator: y的刻度,放大，缩小刻度
    :param title: 标题
    :return:
    '''

    cmap = cm.get_cmap('rainbow',1000)
    figure=plt.figure(facecolor='w')
    #axs保存ax对象
    axs = []
    ax_num = mfrow[0] * mfrow[1]
    shape_map = np.shape(datas[0])

    if xlabels is None:
        xlabels = []
    if ylabels is None:
        ylabels = []

    print('The shape of heatmap：',shape_map)
    #设置x，y坐标轴的刻度等信息
    if draw_axlabel:
        if len(xlabels)==0:
            xlabels = range(0,shape_map[1])
        if len(ylabels)==0:
            ylabels = range(0, shape_map[0])
        #x为shape的1，80*60 为80行60列
        #设置刻度，locator可以控制放大，缩小刻度，调整使得文本能够放下，放置钉子
        xticks = list(range(0,shape_map[1],xlocator))
        yticks = list(range(0,shape_map[0],ylocator))
        #把最后一个加上，用range 中用step不会增加最后一个的吧。。。
        if shape_map[1]-1 not in xticks:
            xticks.append(shape_map[1]-1)
        if shape_map[0]-1 not in yticks:
            yticks.append(shape_map[0]-1)

        xlabels_draw = [xlabels[ind] for ind in xticks]
        ylabels_draw = [ylabels[ind] for ind in yticks]
    # 绘制每个子图的图形
    for i in range(ax_num):
        axs.append(figure.add_subplot(mfrow[0], mfrow[1], i + 1))
        if draw_axlabel:
            axs[i].set_xticks(xticks)
            axs[i].set_xticklabels(xlabels_draw)
            axs[i].set_yticks(yticks )
            axs[i].set_yticklabels(ylabels_draw)
        if len(title)!=0:
            axs[i].set_title('The heatmap of %s'%title[i])
        vmax=datas[i][0][0]
        vmin=datas[i][0][0]
        for row_i in datas[i]:
            for j in row_i:
                if j>vmax:
                    vmax=j
                if j<vmin:
                    vmin=j
        map=axs[i].imshow(datas[i],interpolation='nearest',cmap=cmap,aspect='auto',vmin=vmin,vmax=vmax)
        plt.colorbar(mappable=map,cax=None,ax=None,shrink=0.5)
    plt.show()


def change_series_to_grid(data,shapes):
    #将Series数据变成制定形状的形式
    #为啥不用s.values.reshape()????
    '''
    :param data: pd.Series
    :param shapes: tuple，shape of the gird
    :return: data grid like
    '''
    if len(data) != shapes[0]*shapes[1]:
        print('The lenght of data should be the same as the grid')
        return None
    data_grid= np.ones(shapes)
    i_count = 0
    # 按行生成矩阵
    for i in range(shapes[0]):
        for j in range(shapes[1]):
            data_grid[i, j] =  data[i_count]
            i_count = i_count + 1
    return data_grid


def heatmap_from_dataframe(df,shape_map,mfrow,draw_axlabel=False,
                                 xlabels=None, ylabels=None,draw_title=True,xlocator=1,ylocator=1,title=list()):
    '''
    :param df: pd.DataFrame，也可以是pd.Series
    :param shape_map: the shape of heatmap ---- [row_length, col_length]
    :param mfrow: 图的划分,比如(2,2)
    :param draw_axlabel:
    :param xlabels: x轴的标签
    :param ylabels: y轴的标签
    :param xlocator: x的刻度,用来调整坐标刻度,如果刻度太密，设置为较大的值
    :param ylocator: y的刻度,用来调整坐标刻度
    :param draw_title: 默认为绘制
    :param title: 不指定默认为绘制Dataframe的columns，对于Series可以用[]制定title
    :return:
    '''


    data_to_draw = []
    num_map = mfrow[0] * mfrow[1]

    if xlabels is None:
        xlabels = []
    if ylabels is None:
        ylabels = []
    #判断是否绘制坐标轴的标签
    if not draw_axlabel:
        xlabels = []
        ylabels = []
    #判断是否为Series
    if isinstance(df,pd.Series):
        data_to_draw.append(change_series_to_grid(df,shape_map))
        if draw_title:
            if len(title) == 0:
                title = list([df.name])
        else:
            title = []
    else:
        #如果是Dataframe
        for i_map in range(num_map):
            eachdata = df.ix[:, i_map]
            # 调用方法尝试将series转化为grid
            data_mat = change_series_to_grid(eachdata,shape_map)
            data_to_draw.append(data_mat)
            #判断是否绘制title,2个参数都不指定的话，默认绘制df的columns
            if draw_title:
                if len(title)==0:
                    title = list(df.columns)
            else:
                title=[]
    #绘制还是调用Draw_Heatmap2的办法
    draw_heatmap(data_to_draw,mfrow,draw_axlabel=draw_axlabel,xlabels=xlabels,ylabels=ylabels,
                      xlocator=xlocator,ylocator=ylocator,title=title)