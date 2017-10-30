# -*- coding: utf-8 -*-
"""
Created on Mon Oct 23 12:18:41 2017

define the role of a node: B Z P

@author: Qian Pan
"""

import pandas as pd
import numpy as np
import networkx as nx

def unique(lst):
    '''
    用于统计Series中每个元素出现的次数
    '''
    return dict(zip(*np.unique(lst, return_counts=True)))
    
        
def featrue_normalization(dict_unorm):
    '''
    :param: dict_unorm，未量钢化的数据，字典
    :return: series_normalized，量纲处理后的数据，pd.Series
    
    对数据进行量钢化处理，
    量纲采取z-score规范化，即y = (x-X的平均值)/X的标准差，
    这里的标准差采用无偏样本标准差。
    '''
    std = np.std(list(dict_unorm.values()), ddof = 1)  # ddof = 1,无偏样本标准差
    ave = np.mean(list(dict_unorm.values()))
    dict_normalized = {}
    for key in dict_unorm.keys():
        dict_normalized[key] = (dict_unorm[key] - ave) / std

    return dict_normalized

def node_role(edgedata_community, nodedata, N, node_save_path='Nodes_BZP_result.csv ', community_save_path='Community_result.csv'):
    '''
    :param edgedata_community: 带有社团信息的边数据
    :param nodedata: 节点数据
    :N: 划分出的社团数，社团编号从0开始
    :node_save_path: 节点信息保存地址
    :community_save_path: 社团信息保存地址
    :return: 
        
    对社团划分后的网络进行分析，
    可得到Community_result.csv和Nodes_BZP_result.csv 
    
    针对网络中的每个节点，主要计算其BZP参数值；
    Z: 每个节点在它所在的局部社团的内部连边数intra_communtiy_degree；
    即每个节点在其社团内部连接的重要程度，并且与同社区内的节点进行z-score标准化；
    
    P: 社团内节点的连边在每个社团间的分布情况，取值在[0, 1]；
    0表示该节点所有边都分布在其所在社团内部，1表示该节点所有连边均匀的分布在各个社团内；
    
    B: 是我们自己定义的一个指标，是相对于Z而言，每个节点在它所在的社团的外部连接情况；
    表达式为，B = 节点的跨社团连边数 / 其所在社团连接的总的外部节点数
    
    关于ZP参数计算公式的参考文献：
    [1] Guimerà R, Nunes Amaral LA. Functional cartography of complex metabolic networks.[J].
    Nature, 2005, 433(7028):895.
    '''
    # 对网络进行结构化分析，网络为全连通、无权、无向的网络
    graph = nx.from_pandas_dataframe(edgedata_community, 'Source', 'Target', create_using=nx.Graph())
    G_size = len(graph)

    if graph.number_of_nodes() < 1:
        return pd.DataFrame()
    
    # 用字典存储节点的所有特征信息BZP等参数
    node_features = {}
    
    # 用字典存储每个社团的基本信息
    community_features = {}
    
    # community 属性，包括社团编号、社团大小、社团内部连边数、跨社团连边数、
    # 本社团连接外部港口数、社团内部密度、社团外部密度，全部定义为字典
    dict_community_no = {}
    dict_community_size = {}
    dict_num_intra_community_links = {}
    dict_num_extra_community_links = {}
    dict_extra_community_ports = {}
    dict_internal_density = {}
    dict_external_density = {}
    dict_IO = {}
        
    for community in range(N):
        dict_community_no[community] = community
        list_intra_source = []
        list_intra_target = []
        list_extra_source = []
        list_source = []
        list_extra_target = []
        list_target = []
                
        community_index = nodedata['community'] == community
        list_nodes = nodedata.loc[community_index,'id'].values  # 社团内节点
        dict_community_size[community] = len(list_nodes)
        
        # intra community links
        for i in edgedata_community.index:
            if edgedata_community.loc[i,'Source'] in list_nodes and edgedata_community.loc[i,'Target'] in list_nodes:
                list_intra_source.append(edgedata_community.loc[i,'Source'])
                list_intra_target.append(edgedata_community.loc[i,'Target'])
            else:
                continue
        intra_community_edges = pd.Series(list_intra_source, list_intra_target)  # 社团内的边
        intra_community_ports = pd.concat([pd.Series(list_intra_source),pd.Series(list_intra_target)])
        dict_num_intra_community_links[community] = len(intra_community_edges)
        
        dict_intra_community_degree = unique(intra_community_ports)  # 每个节点在社团内的连边数

        # extra community links        
        for i in edgedata_community.index:
            if edgedata_community.loc[i,'Source'] in list_nodes and edgedata_community.loc[i,'Target'] not in list_nodes:
                list_source.append(edgedata_community.loc[i,'Source'])
                list_extra_target.append(edgedata_community.loc[i,'Target'])
            elif edgedata_community.loc[i,'Source'] not in list_nodes and edgedata_community.loc[i,'Target'] in list_nodes:
                list_extra_source.append(edgedata_community.loc[i,'Source'])
                list_target.append(edgedata_community.loc[i,'Target'])
            else:
                continue
        extra_community_edges1 = list(zip(list_source,list_extra_target))
        extra_community_edges2 = list(zip(list_extra_source,list_target))
        extra_community_edges1 = pd.Series(extra_community_edges1)
        extra_community_edges2 = pd.Series(extra_community_edges2)
        dict_num_extra_community_links[community] = len(extra_community_edges1) + len(extra_community_edges2)
        dict_extra_community_ports[community] = len(set(pd.concat([pd.Series(list_extra_source),pd.Series(list_extra_target)])))
        
        # 每个节点连接的社团外节点数,用于计算P值
        dict_extra_community_degree = {}
        dict_p = {}
        dict_id = {}
        dict_community = {}
        for node in list_nodes:
            dict_community[node] = community
            dict_id[node] = node
            dict_extra_community_degree[node] = nx.degree(graph,node) - dict_intra_community_degree[node]  # 用于求B

            # P
            node_index1 = edgedata_community['Source'] == node
            node_index2 = edgedata_community['Target'] == node
            source_community = edgedata_community.loc[node_index1,'community_Target']
            target_community = edgedata_community.loc[node_index2,'community_Source']
            edge_community = pd.concat([source_community,target_community])
            dict_x = unique(edge_community)
            sum_x = 0           
            for key in dict_x.keys():
                sum_x += (dict_x[key]/ nx.degree(graph,node))**2
            dict_p[node] = 1-sum_x
                
            # B
            dict_b = {}
            for key in dict_extra_community_degree.keys():
                dict_b[key] = dict_extra_community_degree[key] / dict_extra_community_ports[community]
            dict_b = featrue_normalization(dict_b)
                              
            # Z       
            std = np.std(list(dict_intra_community_degree.values()), ddof = 1)  # 无偏样本标准差
            ave = np.average(list(dict_intra_community_degree.values()))
            dict_z = {}
            for key in dict_intra_community_degree.keys():
                dict_z[key] = (dict_intra_community_degree[key] - ave) / std
        
        # 存储过程，用DataFrame
        Community = pd.Series(dict_community)
        B = pd.Series(dict_b)  
        P = pd.Series(dict_p)
        Z = pd.Series(dict_z)
        Id = pd.Series(dict_id)
        intra_community_degree = pd.Series(dict_intra_community_degree)
        node_features = pd.concat([Id,B,Z,P,intra_community_degree,Community],axis=1)
        if node_save_path is not None:
            node_features.to_csv(node_save_path,mode='a',index=False,header=None)
            print('Community %d Nodes File Saved : '%community, node_save_path)

    for key in dict_num_intra_community_links.keys():
        dict_internal_density[key] = 2 * dict_num_intra_community_links[key] / (dict_community_size[key]*(dict_community_size[key]-1))
        dict_external_density[key] = dict_num_extra_community_links[key] / (dict_community_size[key]*(G_size-dict_community_size[key]))
        dict_IO[key] = dict_internal_density[key] / dict_external_density[key]
    
    community_features['CommunityNo'] = dict_community_no
    community_features['CommunitySize'] = dict_community_size  # 社团规模，所含节点数量
    community_features['No.ExternalPorts'] = dict_extra_community_ports  # 社团连接外部港口数
    community_features['IntraCommunityLinks'] = dict_num_intra_community_links
    community_features['InternalDensity'] = dict_internal_density  # 社团内部密度
    community_features['ExternalDensity'] = dict_external_density  # 社团外部密度
    community_features['IO'] = dict_IO  # InternalDensity / ExternalDensity
    
    community_features = pd.DataFrame(community_features)
    
    if community_save_path is not None:
        community_features.to_csv(community_save_path,index=False,header=True)
        print('Community File Saved : ', community_save_path)
    
    return community_features,node_features

# ------------------- examples --------------------------
def main_example():
    # 边数据 - Source - Target - Weight
    edges = [(0,1,5),
             (1,2,6),
             (2,1,3),
             (0,2,4),
             (0,4,1),
             (4,5,1),
             (5,4,5)]
    
    nodedata = pd.DataFrame({'id':[0,1,2,4,5],'community':[0,0,0,1,1]})

    edgedata = pd.DataFrame(edges,columns=['Source','Target','weight'])
    
    edgedata_community = pd.merge(edgedata,nodedata,how='left',left_on='Source',right_on='id')
    edgedata_community = edgedata_community.merge(nodedata,how='left',left_on='Target',right_on='id',suffixes=('_Source', '_Target'))

    community_features,node_features = node_role(edgedata_community, nodedata, 2)

if __name__ == '__main__':
    main_example()

"""
if __name__ == '__main__':
    nodedata = pd.read_csv("0.472-7Nodes.csv",header=0)
    edgedata = pd.read_csv("0.472-7Edges.csv",header=0)
    #G = nx.from_pandas_dataframe(edgedata, 'Source', 'Target', create_using=nx.Graph())
    edgedata_community = ST_community(edgedata,nodedata)
    community_features,node_features = node_role(edgedata_community, nodedata, 7)
"""  