#-*- coding:utf-8 -*-

'''
目的：
    用于对复杂网络相关的分析，创建网络，计算网络特征等
    结合了networkx, igraph, pygrahistry的接口

方法：
    * 从边数据生成网络 - get_graph_from_edgedata
    * 从边数据获取节点 - get_nodes_from_edgedata
    * 将有向边转化为无向边 - as_undirected_edgedata
    * 计算网络的特征 - calculate_graph_features
    * 计算节点的特征 - calculate_node_features
    * 根据度来过滤网络 - degree_filter
    * 计算模块度 - modularity
    * 社区发现 - community_detect
    * 绘制网络 - draw_graph

主要数据：
    edgedata:
        DataFrame;
        网络中边的信息，包含[Source,Target,Weight]信息,也可以是没有权重的

    cluster_result
        DataFrame;
        社区划分的结果，形式为['Id','modularity_class']
        gephi导出的结果为['id','modularity_class']，注意

备注：
    * 2017.10.10
        - 增加计算模块度方法和社区发现方法
        - 增加 networkx2pandas方法

    * 2017.10.17 - 增加计算节点特征的方法

'''

import pandas as pd
import numpy as np
import graphistry
import igraph
import networkx as nx


class NetworkUnity():
    def __init__(self):
        pass

    @staticmethod
    def as_undirected_edgedata(edgedata):
        #将有向边转化为无向边
        index_droped = []
        edgedata_directed = edgedata.copy()
        for ind in edgedata.index:
            source = edgedata.ix[ind, 'Source']
            target = edgedata.ix[ind, 'Target']
            if ind not in index_droped:
                '''
                如果该边没有被丢弃过--例如之前在A-B这条边时,发现存在B-A,
                那么B-A这条边的index会被记录,之后会被丢弃
                '''
                data_target_as_source = edgedata[edgedata['Source'] == target]
                if len(data_target_as_source) >= 1:
                    if source in data_target_as_source['Target'].values:
                        index_2 = data_target_as_source[data_target_as_source['Target'] == source].index[0]
                        edgedata_directed.ix[ind, 'Weight'] += edgedata_directed.ix[index_2, 'Weight']
                        index_droped.append(index_2)
        #被丢弃的边数据
        # data_droped = edgedata.ix[index_droped,:]
        edgedata_directed.drop(index_droped, axis=0, inplace=True)
        return edgedata_directed

    @staticmethod
    def networkx2pandas(graph):
        '''
        :param graph: networkx.Graph/DiGraph
        :return: edgedata, DataFrame
        '''
        def _getedges(g):
            for es in list(g.edges(data=True)):
                yield dict({'Source': es[0], 'Target': es[1]}, **es[2])
        edgedata = pd.DataFrame(_getedges(graph))
        return edgedata

    @staticmethod
    def get_nodes_from_edgedata(edgedata,return_df=True):
        '''
        :param edgedata: 边的数据
        :param return_df: 是否返回Series，默认True，否则为list
        :return: 节点数据
        '''
        source = set(edgedata['Source'])
        target = set(edgedata['Target'])
        nodes = list(source.union(target))
        if return_df:
            nodes = pd.DataFrame(nodes,columns=['Id'])
        return nodes

    @staticmethod
    def get_graph_from_edgedata(edgedata, attr='Weight', directed=True,connected_component=False):
        '''
        :param edgedata: 边的数据
        :param attr: string 或 list; 边的属性数据，如果没有权重，设置attr=None，
        :param directed: 有向图还是无向图
        :param connected_component: 返回最大联通子图，默认为True,对于有向图为weakly_connected
                                    未开发

        :return: networkx.Graph 或 DiGraph
        '''
        if len(edgedata) < 1:
            if directed:
                return nx.DiGraph()
            else:
                return nx.Graph()

        if directed:
            graph = nx.from_pandas_dataframe(edgedata, 'Source', 'Target',
                                             edge_attr=attr, create_using=nx.DiGraph())
            if connected_component:
                #返回最大联通子图
                graph = max(nx.weakly_connected_component_subgraphs(graph), key=len)
        else:
            graph = nx.from_pandas_dataframe(edgedata, 'Source', 'Target',
                                             edge_attr=attr, create_using=nx.Graph())
            if connected_component:
                graph =  max(nx.connected_component_subgraphs(graph), key=len)

        print('Directed Graph ：', graph.is_directed())
        return graph

    @staticmethod
    def calculate_graph_features(graph,centrality=False,save_path=None):
        '''
        :param graph: graph对象,应该是连通的！
        :param centrality: 是否计算中心度信息
        :param save_path: 信息保存地址
        :return: graph的各种网络特征，pd.Series

        用来计算图的各种网络特征，计算时间跟图的大小相关
        大部分特征都是不加权计算的。
        用字典来记载可能更好
        '''
        def _average(node_dic):
            if len(node_dic) < 1:
                return 0
            else:
                return np.average(list(node_dic.values()))
        features = {}
        NODE_NUM = graph.number_of_nodes()

        if NODE_NUM < 1:
            print('Graph is empty')
            return pd.Series()

        features['Node'] = NODE_NUM
        features['Edge'] = graph.number_of_edges()
        features['Density'] = nx.density(graph)
        features['AveDegree'] = _average(graph.degree())
        # features['Diameter'] = nx.diameter(graph)

        # 有向图和无向图
        if not graph.is_directed():
            features['Directed'] = 0
            features['AveClusterCoefficent'] = nx.average_clustering(graph)
            features['AveShortestPathLength'] = nx.average_shortest_path_length(graph)
        else:
            features['Directed'] = 1
            features['AveInDegree'] = _average(graph.in_degree())
            features['AveOutDegree'] = _average(graph.out_degree())

        # 中心性指标
        if centrality:
            # 度中心性
            node_degree_centrality = nx.degree_centrality(graph)
            ave_degree_centrality = _average(node_degree_centrality)
            # 特征向量中心度
            node_eigenvector_centrality = nx.eigenvector_centrality_numpy(graph)
            ave_eigenvector_centrality = _average(node_eigenvector_centrality)
            #介数中心度
            node_betweenness = nx.betweenness_centrality(graph)
            ave_betweenness_centrality = _average(node_betweenness)
            #接近中心度
            node_closeness = nx.closeness_centrality(graph)
            ave_closeness_centrality = _average(node_closeness)

            features['AveDegreeCentrality'] = ave_degree_centrality
            features['AveEigenvectorCentrality'] = ave_eigenvector_centrality
            features['AveBetweennessCentrality'] = ave_betweenness_centrality
            features['AveClosenessCentrality'] = ave_closeness_centrality

        graph_info = pd.Series(features)
        if save_path is not None:
            graph_info.to_csv(save_path,index=True,header=None)
            print('File Saved : ', save_path)

        return graph_info

    @staticmethod
    def calculate_node_features(graph,weight=None,centrality=False, save_path=None):
        '''
        :param graph: networkx.Graph \ Digraph
        :param weight: str, 某些指标是否使用边的权重，weight = 'Weight'
        :param centrality: 是否计算中心性指标
        :param save_path: 保存地址
        :return: DataFrame, node_features
        '''
        if graph.number_of_nodes() < 1:
            return pd.DataFrame()

        features = {}
        features['Degree'] = nx.degree(graph)

        if graph.is_directed():
            features['InDegree'] = graph.in_degree()
            features['OutDegree'] = graph.out_degree()

        if centrality:
            features['DegreeCentrality'] = nx.degree_centrality(graph)
            features['BetweennessCentrality'] = nx.betweenness_centrality(graph)
            features['EigenvectorCentrality'] = nx.eigenvector_centrality_numpy(graph)
            features['ClosenessCentrality'] = nx.closeness_centrality(graph)

        if weight is not None:

            features['WeightedDegree'] = nx.degree(graph,weight=weight)
            if graph.is_directed():
                features['WeightedInDegree'] = graph.in_degree(weight=weight)
                features['WeightedOutDegree'] = graph.out_degree(weight=weight)

            if centrality:
                features['WeightedBetweennessCentrality'] = nx.betweenness_centrality(graph,weight=weight)
                features['WeightedEigenvectorCentrality'] = nx.eigenvector_centrality_numpy(graph,weight=weight)

        node_features = pd.DataFrame(features)
        node_features['Id'] = node_features.index

        if save_path is not None:
            node_features.to_csv(save_path,header=None)
            print('File Saved : ', save_path)

        return node_features

    @staticmethod
    def degree_filter(graph,lower=None,upper=None):
        '''
        :param graph: Networkx.Graph/DiGraph
        :param lower: int/float，the lower limitation of degree
        :param upper: int/float，the upper limitation of degree
        :return: graph after filter
        '''
        node_degree = graph.degree()
        nodes_all = list(graph.nodes())
        print('Node num: ',graph.number_of_nodes())

        data = pd.DataFrame(list(node_degree.items()),columns=['Id','Degree'])

        if lower is not None:
            data = data[data['Degree'] >= lower]
        if upper is not None:
            data = data[data['Degree'] <= upper]

        nodes_saved = list(data['Id'])
        nodes_drop = set(nodes_all).difference(nodes_saved)
        graph.remove_nodes_from(nodes_drop)
        print('Node num: ',graph.number_of_nodes())

        return graph

    @staticmethod
    def draw_graph(graph,nodes=None):
        '''
        采用pygraphistry 来绘制网络图，节点颜色目前还不能超过12种
        :param graph: networkx.Graph/DiGraph
        :param nodes: DataFrame,如果需要按社区颜色绘制，请传入带有社区信息的节点表, ['Id','modulraity_class']
        :return: None
        '''
        graphistry.register(key='contact pygraphistry for api key')

        ploter = graphistry.bind(source='Source', destination='Target').graph(graph)
        if nodes is not None:
            ploter = ploter.bind(node='Id', point_color='modularity_class').nodes(nodes)
        ploter.plot()
        return None

    @staticmethod
    def community_detect(graph=None,edgedata=None,directed=True,
                         use_method=1, use_weight=None):
        '''
        :param edgedata: DataFrame, 边的数据
        :param graph: Networkx.Graph/DiGraph，与edgedata给定一个就行
        :param directed: Bool, 是否有向
        :param use_method: Int, 使用方法
        :param weight_name: String, 社区发现算法是否使用边权重，如果使用,例如weight_name='Weight'
        :return: 带有社区信息的节点表格
        '''
        #创建igraph.Graph类
        if graph is None and edgedata is not None:
            graph = NetworkUnity.get_graph_from_edgedata(edgedata,
                                                         directed=directed,connected_component=True)

        gr = graphistry.bind(source='Source', destination='Target', node='Id', edge_weight='Weight').graph(graph)
        edgedata = NetworkUnity.networkx2pandas(graph)
        ig = gr.pandas2igraph(edgedata, directed=directed)

        #--------------------------------------------------------
        #如果使用边的数据edgedata
        # gr = graphistry.bind(source='Source',destination='Target',edge_weight='Weight').edges(edgedata)
        # nodes = NetworkUnity.get_nodes_from_edgedata(edgedata)
        # gr = gr.bind(node='Id').nodes(nodes)
        # ig = gr.pandas2igraph(edgedata,directed=directed)
        # --------------------------------------------------------

        '''
        关于聚类的方法，
        参考http://pythonhosted.org/python-igraph/igraph.Graph-class.html
        希望以后可以对每一个算法添加一些简单的介绍
        '''

        method_dict = {
            0:'''ig.community_fastgreedy(weights='%s')'''%str(use_weight),
            1:'''ig.community_infomap(edge_weights='%s',trials=10)'''%str(use_weight),
            2:'''ig.community_leading_eigenvector_naive(clusters=10)''',
            3:'''ig.community_leading_eigenvector(clusters=10)''',
            4:'''ig.community_label_propagation(weights='%s')'''%str(use_weight),
            5:'''ig.community_multilevel(weights='%s')'''%str(use_weight),
            6:'''ig.community_optimal_modularity()''',
            7:'''ig.community_edge_betweenness()''',
            8:'''ig.community_spinglass()''',
            }

        detect_method = method_dict.get(use_method)

        if use_weight is None:
            #如果为None,需要把公式里面的冒号去掉,注意如果有多个None,这个方法需要重新写
            detect_method= detect_method.replace('\'','')

        # -------------开始实施社区发现算法-----------
        print('社区发现方法： ',detect_method)
        res_community = eval(detect_method)
        #将社区信息保存到节点信息中
        ig.vs['modulraity_class'] = res_community.membership
        #将节点信息转化为Dataframe表
        edgedata_,nodedata = gr.igraph2pandas(ig)
        modularity = res_community.modularity

        print(res_community.summary())
        print('community size:\n', res_community.sizes())
        print('modularity:\n', modularity)

        return nodedata

    @staticmethod
    def modularity(cluster_result,edgedata=None,graph=None,
                       directed=True, edge_weight='Weight'):
        '''
        :param cluster_result: 聚类结果，参考gephi输出的表，[Id,modulraity_class]
        :param edgedata: 边数据，与graph给定其中一个
        :param graph: networkx中的Graph/DiGraph
        :param directed: 是否为有向图
        :param edge_weight:
            None/str, 计算模块度是否使用边的权重，如果使用，给定边权重的name
            例如edge_weight='Weight'
            如果不使用，请给定为None
        :return: Q值
        ps：
            1.edgedata 和 graph 至少要给定一个
            2.与gephi中计算的模块度结果已经对比过了，结果一致
        '''
        if edgedata is None and graph is not None:
            edgedata = NetworkUnity.networkx2pandas(graph)

        gr = graphistry.bind(source='Source', destination='Target',
                             node='Id',edge_weight=edge_weight)

        ig = gr.pandas2igraph(edgedata,directed=directed)
        nodes = pd.DataFrame(list(ig.vs['Id']), columns=['Id'])
        community_data = pd.merge(nodes, cluster_result, left_on='Id', right_on='Id', how='left')

        if edge_weight is None:
            Q = ig.modularity(list(community_data['modularity_class']),weights=None)
        else:
            Q = ig.modularity(list(community_data['modularity_class']),weights=list(ig.es[edge_weight]))
        return Q


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

    cluster_result = pd.DataFrame({'Id':[0,1,2,4,5],'modularity_class':[0,0,0,1,1]})

    edgedata = pd.DataFrame(edges,columns=['Source','Target','Weight'])
    print('------------Edgedata------------------')
    print(edgedata)

    graph = NetworkUnity.get_graph_from_edgedata(edgedata,
                                                 attr='Weight',
                                                 directed=True,
                                                 connected_component=True)
    # 根据度过滤网络
    # NetworkUnity.degree_filter(graph,lower=0,upper=10)

    # 计算节点特征
    node_info = NetworkUnity.calculate_node_features(graph,weight='Weight',centrality=True)
    print('----------------Node Feature-------------')
    print(node_info.head())

    #计算网络特征
    graph_info = NetworkUnity.calculate_graph_features(graph,centrality=True)
    print('----------------Graph Feature-------------')
    print(graph_info)

    #计算模块都
    q = NetworkUnity.modularity(cluster_result,
                                graph=graph,
                                directed=True,
                                edge_weight='Weight')
    print('---------------- Modularity -------------')
    print(q)

    #社区发现
    print('---------------- Community dectect -------------')
    nodedata_1 = NetworkUnity.community_detect(graph, use_method=1, use_weight=None)
    print(nodedata_1)



if __name__ == '__main__':
    main_example()



