from networkx.algorithms.shortest_paths import weighted
from networkx.algorithms.shortest_paths.generic import average_shortest_path_length
from sqlalchemy import true
from setting import *
from useful_class import *
import os
import time
import math
import multiprocessing
import igraph
import numpy as np
import pandas as pd
import networkx as nx
from networkx.algorithms import community
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.patches import FancyArrowPatch, Ellipse
from matplotlib.colors import ListedColormap, BoundaryNorm
import matplotlib.patheffects as PathEffects
from adjustText import adjust_text
import cartopy.crs as ccrs
import tigramite.data_processing as pp
from tigramite.pcmci import PCMCI
from tigramite.independence_tests import ParCorr
from cartopy.io.shapereader import Reader
from cartopy.feature import ShapelyFeature
from cartopy.mpl.patch import geos_to_path
import cartopy.feature as cfeature
import matplotlib.patches as patches
from geopy.distance import geodesic, lonlat
import matplotlib.collections as collections
# output call graph
from pycallgraph import PyCallGraph
from pycallgraph.output import GraphvizOutput
from pycallgraph import Config
from pycallgraph import GlobbingFilter

import self_group_net_analysis

plt.rc('font', family='Arial')

# Input Data Var
VAR_TARGET = []

VARS_LIST_Climate = ['CO2','VPD', 'Temp', 'Prcp']
VARS_LIST_Water = ['WaterS', 'WaterG']
VARS_LIST_Investment = ['RuralE', 'AgriM', 'Fert']
VARS_LIST_Harvest = ['CropG', 'FruitG', 'IncomeR']

VARS_LIST = VAR_TARGET + VARS_LIST_Climate + VARS_LIST_Water + VARS_LIST_Investment + VARS_LIST_Harvest

# SELF_GA_GROUP_DICT = {
#     'GA_Centroid_Grass_gt_0':'#f57b7a',
#     'GA_Centroid_Grass_lt_0':'#7ab6f5',
#     'GA_Centroid_RuralP_gt_0':'#f57b7a',
#     'GA_Centroid_RuralP_lt_0':'#7ab6f5',
#     'GA_Centroid_Grain_gt_3000':'#f57b7a',
#     'GA_Centroid_Grain_lt_3000':'#7ab6f5',
#     'GA_Centroid_GDP_lt_18060': '#2892c7',
#     'GA_Centroid_GDP_18060-38332': '#bfd38b',
#     'GA_Centroid_GDP_38332-88612': '#fcb344',
#     'GA_Centroid_GDP_gt_88612': '#e71514',
#     'GA_Centroid_AgriM_lt_025': '#2892c7',
#     'GA_Centroid_AgriM_025-056': '#bfd38b',
#     'GA_Centroid_AgriM_056-105': '#fcb344',
#     'GA_Centroid_AgriM_gt_105': '#e71514',
# }

# The time scale of vars
VARS_TIME_SCALE_DICT = {
    'CO2': 'yearly',
    'VPD': 'yearly',
    'Temp': 'yearly',
    'Prcp': 'yearly',
    'WaterS': 'yearly',
    'WaterG': 'yearly',
    'RuralE': 'yearly',
    'AgriM': 'yearly',
    'Fert': 'yearly',
    'CropG': 'yearly',
    'FruitG': 'yearly',
    'IncomeR': 'yearly'
}

# VARS_TIME_SCALE_DICT = {
#     'CO2': 'monthly_yearly',
#     'Temp': 'monthly_yearly',
#     'VPD': 'monthly_yearly',
#     'Prcp': 'monthly_yearly',
#     'WaterS': 'monthly_yearly',
#     'WaterG': 'monthly_yearly',
#     'RuralE': 'yearly',
#     'AgriM': 'yearly',
#     'Fert': 'yearly',
#     'CropG': 'yearly',
#     'FruitG': 'yearly',
#     'IncomeR': 'yearly'
# }

# average betweenness of two subnet
VAR_SIZE_DICT = {
    'CO2': 0.1,
    'VPD': 1.130886137,
    'Temp': 0.403967986,
    'Prcp': 0.954210337,
    'WaterS': 2.083227029,
    'WaterG': 1.452831326,
    'RuralE': 0.61820207,
    'AgriM': 0.554104301,
    'Fert': 0.885279016,
    'CropG': 1.032024945,
    'FruitG': 1.179016966,
    'IncomeR': 0.706249885
}
VAR_COLOR_DICT = {
    'CO2': '#bed742',
    'VPD': '#bed742',
    'Temp': '#f87ba8',
    'Prcp': '#57c9eb',
    'WaterS': '#0aefff',
    'WaterG': '#0c55f3',
    'RuralE': '#6a6da9',
    'AgriM': '#d93a49',
    'Fert': '#b4b4b4',
    'CropG': '#ffc20e',
    'FruitG': '#2fff6d',
    'IncomeR': '#ff9100'
}

VAR_COLOR_DICT_GROUP = {
    'CO2': '#878787',
    'VPD': '#878787',
    'Temp': '#878787',
    'Prcp': '#878787',
    'WaterS': '#878787',
    'WaterG': '#878787',
    'RuralE': '#878787',
    'AgriM': '#878787',
    'Fert': '#878787',
    'CropG': '#878787',
    'FruitG': '#878787',
    'IncomeR': '#878787'
}

VAR_LABEL_DICT = {
    'CO2': 'CO2',
    'VPD': 'VPD',
    'Temp': 'Temp',
    'Prcp': 'Prcp',
    'WaterS': 'WaterS',
    'WaterG': 'WaterG',
    'RuralE': 'RuralE',
    'AgriM': 'AgriM',
    'Fert': 'FertiD',
    'CropG': 'CropY',
    'FruitG': 'FruitY',
    'IncomeR': 'IncomeR'
}

VAR_LABEL_DICT_CN = {
    'VPD': '饱和水汽压差',
    'Temp': '气温',
    'Prcp': '降水',
    'WaterS': '地表水',
    'WaterG': '地下水',
    'RuralE': '农村用电',
    'AgriM': '农业机械',
    'Fert': '化肥用量',
    'CropG': '粮食产量',
    'FruitG': '水果产量',
    'IncomeR': '农民收入'
}

# Load data
# Data like
# GA_ID  Time1       Time2       Time3       Time4 ...
# [int]  [double]    [double]    [double]    [double]
# 00000  0.001       0.002       0.67        1.34
# 00001  0.003       0.022       0.69        2.34
# ...    ...         ...         ...         ...
# This data must have the same index with GA_Centroid

if not os.path.exists(BaseConfig.OUT_PATH + 'InnerNetworkCSV'):
    os.mkdir(BaseConfig.OUT_PATH + 'InnerNetworkCSV')
if not os.path.exists(BaseConfig.OUT_PATH + 'SelfNetworkCSV'):
    os.mkdir(BaseConfig.OUT_PATH + 'SelfNetworkCSV')
if not os.path.exists(BaseConfig.OUT_PATH + 'InnerNetCentralityMap'):
    os.mkdir(BaseConfig.OUT_PATH + 'InnerNetCentralityMap')
if not os.path.exists(BaseConfig.OUT_PATH + 'InnerNetCommunitiesMap'):
    os.mkdir(BaseConfig.OUT_PATH + 'InnerNetCommunitiesMap')
if not os.path.exists(BaseConfig.OUT_PATH + 'SelfNetworkFigs'):
    os.mkdir(BaseConfig.OUT_PATH + 'SelfNetworkFigs')
if not os.path.exists(BaseConfig.OUT_PATH + 'SelfNetworkGrouply'):
    os.mkdir(BaseConfig.OUT_PATH + 'SelfNetworkGrouply')


def build_edges_to_csv():
    """
    build edges from csv data to csv
    """
    build_self_links()
    # # set multiprocessing to get links
    # new_pool = multiprocessing.Pool()
    # jobs = []
    # self_links_p = new_pool.apply_async(build_self_links, args=())
    # jobs.append(self_links_p)
    # print('get_self_links')
    # for var in VARS_LIST:
    #     p = new_pool.apply_async(build_inner_links, args=(var, ))
    #     jobs.append(p)
    #     print('get_inner_links', var)
    # new_pool.close()
    # new_pool.join()


def analyze_nets_to_file():
    """
    analyze these nets
    """
    self_analyze_nets()
    # inner_analyze_nets()


def self_analyze_nets():
    """
    analyze these self nets
    """
    self_edges_df = pd.read_csv(BaseConfig.OUT_PATH +
                                'SelfNetworkCSV//SelfNetworkAll_stretched.csv')
    oriented_edges_df = self_edges_df[(
        self_edges_df['Unoriented'] == 0)].copy()
    # self_draw_net_in_one_and_pos_neg_parts(oriented_edges_df,
    #                                        'SelfNetworkFigs', 'all')
    self_buffer_multiplier_analysis_for_all('SelfNetworkFigs', 'all')
    # self_group_net_analysis.self_draw_group_nets(oriented_edges_df)

    # self_draw_every_nets(oriented_edges_df)
    # vars_weightiest_var_ap = self_draw_links_df_in_linktype(
    #     oriented_edges_df, 'All_edges', '#cce3ff')

    # for var in VARS_LIST:
    #     self_draw_shortest_path_causal_agents_by_var(
    #         oriented_edges_df, vars_weightiest_var_ap[var], var)
    # vars_weightiest_var_ap = [('WaterS', 18.0), ('WaterG', 15.0),
    #                           ('AgriM', 15.0), ('RuralE', 11.0)]
    # self_draw_shortest_path_causal_agents_by_var(oriented_edges_df,
    #                                              vars_weightiest_var_ap,
    #                                              'Grain')

    if VAR_TARGET != []:
        self_draw_links_in_one_has_target_var(oriented_edges_df, VAR_TARGET[0])
        agents_weightiest_var_sp = self_calculate_shortest_path_causal(
            oriented_edges_df, VAR_TARGET[0])
        # agents_weightiest_var_ap = self_calculate_all_path_causal(
        #     filtered_df, VAR_TARGET[0])
        # agents_weightiest_var_sp_pos = self_calculate_shortest_path_causal_pos(
        #     filtered_df, VAR_TARGET[0])
        # agents_weightiest_var_ap_pos = self_calculate_all_path_causal_pos(
        #     filtered_df, VAR_TARGET[0])

        self_draw_agents_weightiest_var(
            agents_weightiest_var_sp,
            'Agents_weightiest_var_sp_to_' + VAR_TARGET[0])
        # self_draw_agents_weightiest_var(agents_weightiest_var_ap,
        #                'Agents_weightiest_var_ap_to_' + VAR_TARGET[0])
        # self_draw_agents_weightiest_var(agents_weightiest_var_sp_pos,
        #                'Agents_weightiest_var_sp_pos_to_' + VAR_TARGET[0])
        # self_draw_agents_weightiest_var(agents_weightiest_var_ap_pos,
        #                'Agents_weightiest_var_ap_pos_to_' + VAR_TARGET[0])


def self_draw_every_nets(p_edges_df):
    centroid_data = pd.read_csv(BaseConfig.GEO_AGENT_PATH + 'GA_Centroid.csv')
    for agent in list(centroid_data['GA_ID']):
        if agent in list(p_edges_df['Source'].unique()):
            agent_self_net = build_net_by_links_df(
                p_edges_df[(p_edges_df['Source'] == agent)
                           & (p_edges_df['Target'] == agent)].copy())
            self_draw_net_for_agent(agent_self_net, agent)


def self_draw_net_in_one_and_pos_neg_parts(p_self_net_df, p_floder, p_group):
    # prepare nodes
    var_sou = p_self_net_df['VarSou'].map(str)
    var_tar = p_self_net_df['VarTar'].map(str)
    all_ver_list = list(p_self_net_df['VarSou']) + list(
        p_self_net_df['VarTar'])
    # set the unique of the vertexs
    ver_list_unique = list(set(all_ver_list))

    # build one net network
    network = nx.MultiDiGraph()
    for v_id_var in ver_list_unique:
        network.add_node(v_id_var,
                         label=VAR_LABEL_DICT[v_id_var],
                         size=30,
                         color=VAR_COLOR_DICT[v_id_var],
                         color_g=VAR_COLOR_DICT_GROUP[v_id_var],
                         label_size=15)
    for lIndex, lRow in p_self_net_df.iterrows():
        thisSou = lRow["VarSou"]
        thisTar = lRow["VarTar"]
        network.add_edge(thisSou,
                         thisTar,
                         weight=abs(lRow['Strength']),
                         color=lRow['Strength'],
                         strength=lRow['Strength'],
                         timelag=abs(lRow['TimeLag']))
    output_net_info_by_nx(
        network, BaseConfig.OUT_PATH + p_floder + '//' + p_group +
        '_Self_oneNet_info.csv', False)
    nodes_pos_in1 = draw_network(
        network, 'network_overall', BaseConfig.OUT_PATH + p_floder + '//' +
        p_group + '_Self_network_overall.pdf')

    # build two net network
    network_pos = nx.MultiDiGraph()
    network_neg = nx.MultiDiGraph()
    for v_id_var in ver_list_unique:
        network_pos.add_node(v_id_var,
                             label=VAR_LABEL_DICT[v_id_var],
                             size=30,
                             color=VAR_COLOR_DICT[v_id_var],
                             color_g=VAR_COLOR_DICT_GROUP[v_id_var],
                             label_size=15)
        network_neg.add_node(v_id_var,
                             label=VAR_LABEL_DICT[v_id_var],
                             size=30,
                             color=VAR_COLOR_DICT[v_id_var],
                             color_g=VAR_COLOR_DICT_GROUP[v_id_var],
                             label_size=15)
    for lIndex, lRow in p_self_net_df.iterrows():
        thisSou = lRow["VarSou"]
        thisTar = lRow["VarTar"]
        if lRow['Strength'] >= 0:
            strength_temp = lRow['Strength']
            if lRow['Strength'] == 0:
                strength_temp = 0.0005
            network_pos.add_edge(thisSou,
                                 thisTar,
                                 weight=abs(lRow['Strength']),
                                 color=lRow['Strength'],
                                 strength=strength_temp,
                                 timelag=abs(lRow['TimeLag']))
        else:
            network_neg.add_edge(thisSou,
                                 thisTar,
                                 weight=abs(lRow['Strength']),
                                 color=lRow['Strength'],
                                 strength=abs(lRow['Strength']),
                                 timelag=abs(lRow['TimeLag']))
    output_net_info_by_nx(
        network_pos, BaseConfig.OUT_PATH + p_floder + '//' + p_group +
        '_Self_network_pos_info.csv', True)
    output_net_info_by_nx(
        network_neg, BaseConfig.OUT_PATH + p_floder + '//' + p_group +
        '_Self_network_neg_info.csv', True)
    draw_network(
        network_pos, 'network_pos', BaseConfig.OUT_PATH + p_floder + '//' +
        p_group + '_Self_network_pos.pdf', nodes_pos_in1)
    draw_network(
        network_neg, 'network_neg', BaseConfig.OUT_PATH + p_floder + '//' +
        p_group + '_Self_network_neg.pdf', nodes_pos_in1)


def draw_network(network, info, path, nodes_=None):
    fig = plt.figure(figsize=(10, 10), dpi=500)
    ax = fig.add_subplot()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    # plt.title('NetWork')

    if nodes_ == None:
        # nodes_ = nx.spring_layout(network, k=0.5, iterations=5)
        nodes_ = nx.nx_pydot.graphviz_layout(network, prog='sfdp', root=None)
        # prog = dot neato fdp sfdp circo twopi osage patchwork

    # set nodes colors
    # community_list = community.asyn_fluidc(nx.MultiGraph(network), 2)
    community_list = community.kernighan_lin.kernighan_lin_bisection(
        nx.MultiGraph(network), weight='strength')
    com_group_number = {}
    color_list = ['#82cc5c', '#95a4f7', '#bcbd22']
    for c_index, com in enumerate(list(community_list)):
        for n in com:
            com_group_number[n] = color_list[c_index]
    node_color = [com_group_number[n] for n in network]

    ns_by_degree = []
    for n in network:
        ns_by_degree.append(network.degree(n) * 5)

    ns_by_betweenness = []
    i_net = igraph.Graph.from_networkx(network)
    betweenness = i_net.betweenness()
    nc = []
    stimulating_buffer = '#ffbd80'
    stimulating_multiplier = '#ff7c67'
    inhibiting_buffer = '#6feadf'
    inhibiting_multiplier = '#5db9f1'
    b_m_fig_info = pd.read_csv(BaseConfig.OUT_PATH + 'SelfNetworkFigs' + '//' +
                               'ready-all_Self_buffer_multiplier_analysis_info.csv')
    if info == 'network_overall':
        all_info_df=pd.read_csv(BaseConfig.OUT_PATH + 'SelfNetworkFigs' + '//' +
                    'ready-all_Self_oneNet_info.csv')
        for n in network:
            inter_info = b_m_fig_info[b_m_fig_info['Node'] == n]
            if inter_info['Interaction effect'].values > 1:
                if inter_info['Activity level of Stimulation'].values > 1:
                    nc.append(stimulating_multiplier)
                else:
                    nc.append(stimulating_buffer)
            else:
                if inter_info['Activity level of Inhibition'].values > 1:
                    nc.append(inhibiting_multiplier)
                else:
                    nc.append(inhibiting_buffer)
            # ns_by_betweenness.append(1000*(all_info_df[all_info_df['Node']==n]['degree'].values[0]))
            ns_by_betweenness.append(1000*VAR_SIZE_DICT[n])

    if info == 'network_pos':
        pos_info_df=pd.read_csv(BaseConfig.OUT_PATH + 'SelfNetworkFigs' + '//' +
                            'ready-all_Self_network_pos_info.csv')
        for n in network:
            nc.append(pos_info_df[pos_info_df['Node']==n]['degree'].values[0])
            ns_by_betweenness.append(1000*(pos_info_df[pos_info_df['Node']==n]['betweenness'].values[0]))
            # ns_by_betweenness.append(1000*bet)
    if info == 'network_neg':
        neg_info_df=pd.read_csv(BaseConfig.OUT_PATH + 'SelfNetworkFigs' + '//' +
                'ready-all_Self_network_neg_info.csv')
        for n in network:
            nc.append(neg_info_df[neg_info_df['Node']==n]['degree'].values[0])
            # nc.append(network.degree(n))
            ns_by_betweenness.append(1000*(neg_info_df[neg_info_df['Node']==n]['betweenness'].values[0]))
            # ns_by_betweenness.append(1000*bet)

    edge_color = []
    for u, v, d in network.edges(data=True):
        color = float(d['color'])
        if color < 0:
            # edge_color.append(cm.Blues(-color))
            edge_color.append('#69acd6')
        else:
            # edge_color.append(cm.Reds(color))
            edge_color.append('#fa6648')

    edge_width = []
    for u, v, d in network.edges(data=True):
        weight = float(d['weight'])
        edge_width.append(weight * 2)

    nx.draw_networkx_edges(network,
                           nodes_,
                           width=edge_width,
                           alpha=0.1,
                           edge_color=edge_color,
                           arrows=True,
                           arrowstyle='->',
                           min_target_margin=20,
                           arrowsize=10,
                           connectionstyle="arc3,rad=0.1")

    nx.draw_networkx_nodes(
        network,
        nodes_,
        node_size=ns_by_betweenness,
        node_color=nc,
        cmap=cm.Wistia,
        # node_color=node_color,
        # node_color=[d['color_g'] for n, d in network.nodes(data=True)],
        alpha=0.8)

    nx.draw_networkx_labels(network,
                            nodes_,
                            labels={
                                n: network.nodes[n]['label']
                                for n, d in network.nodes(data=True)
                            },
                            font_size=14,
                            font_color='k',
                            font_family='Arial',
                            font_weight='bold')
    plt.savefig(path)
    return nodes_


def self_draw_links_in_one_has_target_var(p_self_net_df, p_target_var):
    # build a net network
    network = nx.MultiDiGraph()
    # add every vertex to the net
    var_sou = p_self_net_df['VarSou'].map(str)
    var_tar = p_self_net_df['VarTar'].map(str)
    all_ver_list = list(p_self_net_df['VarSou']) + list(
        p_self_net_df['VarTar'])
    # set the unique of the vertexs
    ver_list_unique = list(set(all_ver_list))
    for v_id_var in ver_list_unique:
        network.add_node(v_id_var,
                         label=VAR_LABEL_DICT[v_id_var],
                         size=30,
                         color=VAR_COLOR_DICT[v_id_var],
                         label_size=15)
    for lIndex, lRow in p_self_net_df.iterrows():
        thisSou = lRow["VarSou"]
        thisTar = lRow["VarTar"]
        network.add_edge(thisSou,
                         thisTar,
                         weight=abs(lRow['Strength']),
                         strength=lRow['Strength'],
                         timelag=abs(lRow['TimeLag']))

    output_net_info_by_nx(
        network, BaseConfig.OUT_PATH +
        'SelfNetworkFigs//Self_info_all_in_one_by_' + p_target_var + '.csv')

    fig = plt.figure(figsize=(10, 10), dpi=500)
    ax = fig.add_subplot()
    mapTitle = 'NetWork'
    plt.title(mapTitle)

    # pos_old = nx.spiral_layout(network)

    pos_old = nx.spiral_layout(network, equidistant=True, resolution=0.35)
    degree_of_nodes = {}
    for var in VARS_LIST:
        if var in pos_old.keys():
            if var != p_target_var:
                degree_of_nodes[var] = network.degree(var)
    degree_of_nodes_sorted = sorted(degree_of_nodes.items(),
                                    key=lambda x: x[1],
                                    reverse=False)
    nodes_order = [t[0] for t in degree_of_nodes_sorted]
    nodes_order = [p_target_var] + nodes_order
    nodes_order.append(p_target_var)
    pos = dict(map(lambda x, y: [x, y], nodes_order, pos_old.values()))
    # first draw point's edge in black to set Point edge
    nx.draw_networkx_nodes(
        network,
        pos,
        node_size=[network.degree(n) * 1.4 for n in network],
        node_color='k',
        alpha=0.7)
    nx.draw_networkx_nodes(
        network,
        pos,
        node_size=[network.degree(n) * 1.3 for n in network],
        node_color=[d['color'] for n, d in network.nodes(data=True)],
        alpha=0.7)

    nx.draw_networkx_edges(network,
                           pos,
                           width=1,
                           alpha=0.35,
                           edge_color=[
                               float(d['strength'])
                               for (u, v, d) in network.edges(data=True)
                           ],
                           edge_cmap=plt.cm.RdBu_r,
                           arrows=True,
                           arrowstyle='->',
                           min_target_margin=16,
                           arrowsize=8,
                           connectionstyle="arc3,rad=0.1")

    nx.draw_networkx_labels(network,
                            pos,
                            labels={
                                n: network.nodes[n]['label']
                                for n, d in network.nodes(data=True)
                            },
                            font_size=14,
                            font_color='k',
                            font_family='Times New Roman',
                            font_weight='bold')

    plt.savefig(BaseConfig.OUT_PATH + 'SelfNetworkFigs//Self_all_in_one_by_' +
                p_target_var + '.pdf')


def self_buffer_multiplier_analysis_for_all(p_data_folder, p_group):
    # read csv
    network_info = pd.read_csv(BaseConfig.OUT_PATH + p_data_folder + '//' +
                               p_group + '_Self_oneNet_info.csv')
    network_pos_info = pd.read_csv(BaseConfig.OUT_PATH + p_data_folder + '//' +
                                   p_group + '_Self_network_pos_info.csv')
    network_neg_info = pd.read_csv(BaseConfig.OUT_PATH + p_data_folder + '//' +
                                   p_group + '_Self_network_neg_info.csv')
    bm_fig_df=pd.read_csv(BaseConfig.OUT_PATH + p_data_folder + '//ready-all_Self_buffer_multiplier_analysis_info.csv')

    fig = plt.figure(figsize=(11, 6), dpi=500)
    rect1 = [0, 0, 1, 1]  # [左, 下, 宽, 高] 规定的矩形区域 （全部是0~1之间的数，表示比例）
    ax1 = plt.axes(rect1)

    ax1.grid(color='#ffffff', zorder=-10, axis='both')
    #left down
    ax1.add_patch(
        patches.Rectangle((0, 0),
                          1,
                          1,
                          facecolor="#7eb9d7",
                          alpha=0.4,
                          clip_on=False))
    #right down
    ax1.add_patch(
        patches.Rectangle((1, 0),
                          1.5,
                          1,
                          facecolor="#f47d6a",
                          alpha=0.4,
                          clip_on=False))
    #left up
    ax1.add_patch(
        patches.Rectangle((0, 1),
                          1,
                          1.5,
                          facecolor="#7eb9d7",
                          alpha=0.8,
                          clip_on=False))
    #right up
    ax1.add_patch(
        patches.Rectangle((1, 1),
                          1.5,
                          1.5,
                          facecolor="#f47d6a",
                          alpha=0.8,
                          clip_on=False))

    # ax1.axvspan(0.3,
    #             1,
    #             facecolor='#bfdaec',
    #             alpha=0.7,
    #             zorder=-10,
    #             clip_on=False)
    ax1.axvline(x=1, linewidth=2, color='#ffffff', linestyle='--', zorder=1)
    ax1.axhline(y=1, linewidth=2, color='#ffffff', linestyle='--', zorder=1)

    sizes = []
    for n in bm_fig_df['Node']:
        sizes.append(1000*VAR_SIZE_DICT[n])

    for axes in [ax1]:
        axes.scatter(bm_fig_df['Interaction effect'] ,
                     bm_fig_df['Driving effect'],
                     marker='o',
                     color='#ffffff',
                     s=sizes,
                     alpha=0.8,
                     edgecolors='#595c5d',
                     linewidths=1,
                     label='  betweenness = 0.6 \n\n  betweenness = 0.3 ',
                     zorder=2)
        for x, y, text in zip(bm_fig_df['Interaction effect'] ,
                     bm_fig_df['Driving effect'], bm_fig_df['Node']):
            ax1.text(x,
                     y,
                     VAR_LABEL_DICT[text],
                     fontsize=14,
                     ha='center',
                     va='center',
                     color='#000000')

    ax1.set_xlim(0, 2.5)
    ax1.set_ylim(0, 2.5)

    ax1.set_xticks([0, 0.5, 1, 1.5, 2,2.5])
    label_X1 = ['0', '0.5', '1', '1.5', '2','2.5']
    ax1.set_xticklabels(label_X1, fontsize=14, fontfamily='Arial')

    plt.yticks(fontproperties='Arial', size=14)
    plt.xticks(fontproperties='Arial', size=14)
    ax1.set_xlabel(
        'Interaction effect\n(weighted degree in stimulation sub-network / weighted degree in inhibition sub-network)',
        fontsize=16)
    ax1.set_ylabel('Driving effect (overall outdegree / overall indegree)',
                   fontsize=16)
    ax1.legend(loc='upper left',
               ncol=1,
               prop={
                   'size': 14,
               },
               labelspacing=1,
               borderpad=0.8,
               handletextpad=0.1,
               markerscale=0.0000000000001,
               frameon=True)
    ax1.scatter([0.1, 0.1], [2.34, 2.19],
                marker='o',
                color=['#ffffff', '#ffffff'],
                s=[0.6 * 1000, 0.3 * 1000],
                alpha=0.8,
                edgecolors='#595c5d',
                linewidths=1,
                zorder=100)

    ax1.text(0.25,
             -0.12,
             '← inhibiting',
             fontsize=16,
             c='#000000',
             style='italic',
             ha='center',
             va='center')
    ax1.text(2.25,
             -0.12,
             'stimulating →',
             fontsize=16,
             c='#000000',
             style='italic',
             ha='center',
             va='center')

    plt.savefig(BaseConfig.OUT_PATH + p_data_folder + '//' + p_group +
                '_Self_buffer_multiplier_analysis_1.pdf',
                dpi=500,
                bbox_inches='tight')

    fig = plt.figure(figsize=(11, 6), dpi=500)
    # ax = fig.add_subplot(1, 1, 1)
    rect = [0, 0, 1, 1]  # [左, 下, 宽, 高] 规定的矩形区域 （全部是0~1之间的数，表示比例）
    ax = plt.axes(rect)

    ax.grid(color='#ffffff', zorder=-10)
    ax.axvspan(0.01, 1, facecolor='#7e968e', alpha=1, zorder=-10)
    ax.axvspan(1, 100, facecolor='#cfe1e8', alpha=1, zorder=-10)
    ax.axvline(x=1, linewidth=2, color='#ffffff', linestyle='--', zorder=1)
    # ax.axvline(x=1, linewidth=2, color='#b2735e', linestyle='--', zorder=-10)

    # x_c = 0.1
    x_c = -0.8
    y_c = 87.5
    plt.text(math.pow(10, x_c),
             y_c + 18,
             'buffers',
             fontsize=26,
             c='#242b31',
             style='italic',
             ha='center',
             va='center')
    G_buffer = nx.DiGraph()
    G_buffer.add_nodes_from([0, 1, 2, 3, 4, 5])
    G_buffer.add_edges_from([(0, 4), (1, 4), (2, 4), (3, 4), (4, 5)])

    pos_buffer = {
        0: np.array([math.pow(10, x_c - 0.18), y_c + 13]),
        1: np.array([math.pow(10, x_c - 0.2), y_c + 5]),
        2: np.array([math.pow(10, x_c - 0.2), y_c - 5]),
        3: np.array([math.pow(10, x_c - 0.18), y_c - 13]),
        4: np.array([math.pow(10, x_c), y_c]),
        5: np.array([math.pow(10, x_c + 0.18), y_c])
    }
    nx.draw_networkx_edges(G_buffer,
                           pos_buffer,
                           ax=ax,
                           width=1,
                           alpha=1,
                           edge_color='k',
                           arrows=True,
                           arrowstyle='->',
                           arrowsize=10,
                           node_size=600)
    nx.draw_networkx_nodes(G_buffer,
                           pos_buffer,
                           ax=ax,
                           nodelist=[4],
                           node_size=700,
                           node_color='#ffffff',
                           edgecolors='#000000',
                           alpha=1)
    ax.set_axis_on()
    ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)

    # x_c = 10
    x_c = 0.8
    y_c = 87.5
    plt.text(math.pow(10, x_c),
             y_c + 18,
             'multipliers',
             fontsize=26,
             c='#242b31',
             style='italic',
             ha='center',
             va='center')
    G_multi = nx.DiGraph()
    G_multi.add_nodes_from([0, 1, 2, 3, 4, 5])
    G_multi.add_edges_from([(0, 1), (1, 2), (1, 3), (1, 4), (1, 5)])

    pos_multi = {
        0: np.array([math.pow(10, x_c - 0.2), y_c]),
        1: np.array([math.pow(10, x_c), y_c]),
        2: np.array([math.pow(10, x_c + 0.18), y_c + 13]),
        3: np.array([math.pow(10, x_c + 0.2), y_c + 5]),
        4: np.array([math.pow(10, x_c + 0.2), y_c - 5]),
        5: np.array([math.pow(10, x_c + 0.18), y_c - 13])
    }
    nx.draw_networkx_edges(G_multi,
                           pos_multi,
                           ax=ax,
                           width=1,
                           alpha=1,
                           edge_color='k',
                           arrows=True,
                           arrowstyle='->',
                           arrowsize=10,
                           node_size=600)
    nx.draw_networkx_nodes(G_multi,
                           pos_multi,
                           ax=ax,
                           nodelist=[1],
                           node_size=700,
                           node_color='#ffffff',
                           edgecolors='#000000',
                           alpha=1)
    ax.set_axis_on()
    ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)

    ax.set_xscale('symlog', linthresh=0.05)
    # ax.set_xscale('log')

    buffer_multiplier_info_df = pd.DataFrame()
    buffer_multiplier_info_df = buffer_multiplier_info_df.append(
        pd.DataFrame({
            'Node':
            network_info['Node'],
            'Interaction effect':
            network_pos_info['degree_weighted'] /
            network_neg_info['degree_weighted'],
            'Driving effect':
            network_info['outdegree_div_indegree'],
            'Activity level of Stimulation':
            network_pos_info['out_div_in_weighted'],
            'Activity level of Inhibition':
            network_neg_info['out_div_in_weighted'],
            'Interconnectedness of Stimulation':
            network_pos_info['degree_weighted'],
            'Interconnectedness of Inhibition':
            network_neg_info['degree_weighted'],
        }),
        ignore_index=False)
    buffer_multiplier_info_df.to_csv(
        BaseConfig.OUT_PATH + p_data_folder + '//' + p_group +
        '_Self_buffer_multiplier_analysis_info.csv')

    network_pos_info = network_pos_info.sort_values(by='betweenness',
                                                    ascending=False)
    sizes = []

    info_df=pd.read_csv(BaseConfig.OUT_PATH + 'SelfNetworkFigs' + '//' +
                            'ready-all_Self_network_pos_info.csv')
    for n in bm_fig_df['Node']:
        bet=info_df[info_df['Node']==n]['betweenness'].values[0]
        if bet < 0.1:
            sizes.append(0.1 * 1000)
        else:
            sizes.append(bet * 1000)

    plt.scatter(bm_fig_df['Activity level of Stimulation'],
                bm_fig_df['Interconnectedness of Stimulation'],
                marker='o',
                color='#fa6648',
                s=sizes,
                alpha=0.8,
                edgecolors='#595c5d',
                linewidths=1,
                label='Stimulation sub-network           betweenness = 0.6 ',
                zorder=2)
    for x, y, text in zip(bm_fig_df['Activity level of Stimulation'],
                bm_fig_df['Interconnectedness of Stimulation'],
                          bm_fig_df['Node']):
        if text == 'FruitG':
            y = y + 1
        if text == 'CropG':
            y = y - 1
        if text == 'WaterS':
            y = y + 2
        if text == 'Prcp':
            x = x + 0.065
        plt.text(x,
                 y,
                 VAR_LABEL_DICT[text],
                 fontsize=14,
                 ha='center',
                 va='center',
                 color='#450208')

    network_neg_info = network_neg_info.sort_values(by='betweenness',
                                                    ascending=False)
    sizes = []
    info_df=pd.read_csv(BaseConfig.OUT_PATH + 'SelfNetworkFigs' + '//' +
                            'ready-all_Self_network_neg_info.csv')
    for n in bm_fig_df['Node']:
        bet=info_df[info_df['Node']==n]['betweenness'].values[0]
        if bet < 0.1:
            sizes.append(0.1 * 1000)
        else:
            sizes.append(bet * 1000)

    plt.scatter(bm_fig_df['Activity level of Inhibition'],
                bm_fig_df['Interconnectedness of Inhibition'],
                marker='o',
                color='#69acd6',
                s=sizes,
                alpha=0.8,
                edgecolors='#595c5d',
                linewidths=1,
                label='Inhibition sub-network              betweenness = 0.3 ',
                zorder=2)
    for x, y, text in zip(bm_fig_df['Activity level of Inhibition'],
                bm_fig_df['Interconnectedness of Inhibition'],
                          bm_fig_df['Node']):
        if text == 'FruitG':
            x = x - 0.05
        if text == 'RuralE':
            x = x - 0.075
        if text == 'AgriM':
            x = x + 0.06
        if text == 'Temp':
            x = x + 0.1
        plt.text(x,
                 y,
                 VAR_LABEL_DICT[text],
                 fontsize=14,
                 ha='center',
                 va='center',
                 color='#042145')

    ax.legend(loc='upper left',
              ncol=1,
              prop={
                  'size': 14,
              },
              labelspacing=1,
              borderpad=0.8,
              handletextpad=0.1,
              markerscale=0.0000000001,
              frameon=True)
    plt.scatter([0.118, 0.118, 0.36, 0.36], [154, 165, 154, 165],
                marker='o',
                color=['#69acd6', '#fa6648', '#ffffff', '#ffffff'],
                s=[0.45 * 1000, 0.45 * 1000, 0.3 * 1000, 0.6 * 1000],
                alpha=0.8,
                edgecolors='#595c5d',
                linewidths=1,
                zorder=100)

    leftlim = math.pow(10, -1)
    rightlim = math.pow(10, 1)
    ax.set_xlim(leftlim, rightlim)
    ax.set_ylim(0, 175)
    ax.set_yticks([0, 25, 50, 75, 100, 125, 150, 175])
    ax.set_xticks([
        math.pow(10, -1),
        math.pow(10, -0.5),
        math.pow(10, 0),
        math.pow(10, 0.5),
        math.pow(10, 1)
    ])
    label_X = ['0.1', '0.316', '1', '3.16', '10']
    ax.set_xticklabels(label_X, fontsize=14, fontfamily='Arial')
    plt.yticks(fontproperties='Arial', size=14)
    plt.xticks(fontproperties='Arial', size=14)
    ax.set_xlabel('Activity level\n(weighted outdegree / weighted indegree)',
                  fontsize=16)
    ax.set_ylabel('Interconnectedness (weighted degree)', fontsize=16)
    plt.text(math.pow(10, -0.75),
             -12,
             '← influenced',
             fontsize=16,
             c='#000000',
             style='italic',
             ha='center',
             va='center')
    plt.text(math.pow(10, 0.75),
             -12,
             'influencing →',
             fontsize=16,
             c='#000000',
             style='italic',
             ha='center',
             va='center')

    plt.savefig(BaseConfig.OUT_PATH + p_data_folder + '//' + p_group +
                '_Self_buffer_multiplier_analysis_2.pdf',
                dpi=500,
                bbox_inches='tight')


def self_draw_links_df_in_linktype(p_self_net_df, p_GA_group_name, p_color):
    p_self_net_df['Count'] = 1
    linktype_counts = p_self_net_df.groupby(['VarSou', 'VarTar'],
                                            as_index=False)['Count'].count()
    linktype_timelags = p_self_net_df.groupby(
        ['VarSou', 'VarTar'], as_index=False)['TimeLag'].mean()
    linktype_strengths = p_self_net_df.groupby(
        ['VarSou', 'VarTar'], as_index=False)['Strength'].mean()
    linksbyType = pd.concat([
        linktype_counts, linktype_timelags['TimeLag'],
        linktype_strengths['Strength']
    ],
                            axis=1)
    linksbyType.to_csv(BaseConfig.OUT_PATH + 'SelfNetworkGrouply//' +
                       'Edges_of_linktype_' + p_GA_group_name + '.csv')
    # build a net network
    network = nx.DiGraph()
    # add every vertex to the net
    all_ver_list = list(linksbyType['VarSou']) + list(linksbyType['VarTar'])
    # set the unique of the vertexs
    ver_list_unique = list(set(all_ver_list))
    for v_id_var in ver_list_unique:
        network.add_node(v_id_var,
                         label=VAR_LABEL_DICT[v_id_var],
                         size=30,
                         color=VAR_COLOR_DICT[v_id_var],
                         label_size=15)
    for lIndex, lRow in linksbyType.iterrows():
        thisSou = lRow["VarSou"]
        thisTar = lRow["VarTar"]
        network.add_edge(thisSou,
                         thisTar,
                         weight=lRow['Count'],
                         strength=lRow['Strength'])
    output_net_info_by_nx(
        network, BaseConfig.OUT_PATH +
        'SelfNetworkGrouply//Net_info_of_linktype_' + p_GA_group_name + '.csv')
    # communities = community.kernighan_lin_bisection(network)
    # print(communities)
    # community_list = community.asyn_fluidc(nx.MultiGraph(network), 2)
    # com_group_number = {}
    # for c_index, com in enumerate(list(community_list)):
    #     for n in com:
    #         com_group_number[n] = c_index
    # # compute centrality
    centrality = nx.betweenness_centrality(nx.Graph(network),
                                           k=10,
                                           endpoints=True)

    #### draw graph ####
    fig, ax = plt.subplots(figsize=(10, 10))
    # pos = nx.spring_layout(network, k=0.05, weight='weight', seed=1)
    pos = nx.circular_layout(network)
    # node_color = [com_group_number[n] for n in network]
    node_size = [v * 20000 for v in centrality.values()]
    # edge_color = [
    #     float(d['strength']) for (u, v, d) in network.edges(data=True)
    # ]

    # cmap = plt.get_cmap('coolwarm')
    # norm = matplotlib.colors.Normalize(vmin=-1,
    #                                    vmax=1)
    # sm = plt.cm.ScalarMappable(cmap=cmap,norm=norm)

    edge_color = []
    for u, v, d in network.edges(data=True):
        strength = float(d['strength'])
        if strength < 0:
            edge_color.append(cm.Blues(-strength))
        else:
            edge_color.append(cm.Reds(strength))

    edge_width_v = [
        float(d['weight']) for (u, v, d) in network.edges(data=True)
    ]
    edge_width = 2 + 3 * (np.array(edge_width_v) - np.array(edge_width_v).min(
    )) / (np.array(edge_width_v).max() - np.array(edge_width_v).min())
    # edge_width = []
    # for u, v, d in network.edges(data=True):
    #     width = float(d['weight']) * 1.6
    #     if width > 5:
    #         edge_width.append(5)
    #     else:
    #         edge_width.append(width)
    # nx.draw_networkx(
    #     network,
    #     pos=pos,
    #     with_labels=False,
    #     node_color='#9ab99a',
    #     node_size=node_size,
    #     edge_color=edge_color,
    #     width=edge_width,
    #     edge_cmap=plt.cm.coolwarm,
    #     alpha=0.5,
    # )
    nx.draw_networkx_nodes(
        network,
        pos,
        node_size=node_size,
        # node_color=[network.degree(n)  for n in network],
        node_color=p_color,
        # cmap=plt.cm.Wistia,
        edgecolors='#d9d9d9',
        alpha=0.85)

    nx.draw_networkx_edges(
        network,
        pos,
        width=edge_width,
        alpha=0.6,
        edge_color=edge_color,
        #    edge_cmap=plt.cm.coolwarm,
        arrows=True,
        arrowstyle='->',
        min_target_margin=30,
        arrowsize=8,
        connectionstyle="arc3,rad=0.1")

    nx.draw_networkx_labels(
        network,
        pos=pos,
        labels={
            n: network.nodes[n]['label']
            for n, d in network.nodes(data=True)
        },
        font_size=14,
        font_color='k',
        # font_family='SimHei',
        font_family='Times New Roman',
        font_weight='bold')
    fig.tight_layout()
    plt.axis("off")
    plt.savefig(BaseConfig.OUT_PATH + 'SelfNetworkGrouply//Linktype_of_' +
                p_GA_group_name + '.pdf')
    self_calculate_indirect_strength_in_linktype(network, p_GA_group_name)
    return self_calculate_most_edges_in_linktype(network)


def self_calculate_indirect_strength_in_linktype(p_linktypenet, p_name):
    indirect_info_df = pd.DataFrame(columns=['VarSou', 'VarTar'])
    for target_var in VARS_LIST:
        for var in VARS_LIST:
            if var == target_var:
                continue
            if not var in p_linktypenet.nodes():
                continue
            try:
                all_paths = nx.all_simple_paths(p_linktypenet, var, target_var,
                                                4)
            except:
                continue
            strength_sum = 0
            paths_count = 0
            for a_path in all_paths:
                a_path_strength = 0
                for i, i_var in enumerate(a_path):
                    if i == len(a_path) - 1:
                        break
                    props = p_linktypenet[a_path[i]][a_path[i + 1]]
                    a_path_strength = a_path_strength + props['strength']
                a_path_strength_mean = a_path_strength / (len(a_path) - 1)
                strength_sum = strength_sum + a_path_strength_mean
                paths_count = paths_count + 1
            if paths_count == 0:
                continue
            indirect_info_df = indirect_info_df.append(pd.DataFrame({
                'VarSou': [var],
                'VarTar': [target_var],
                'Indirect_Strength_Sum': [strength_sum],
                'Paths_Count': [paths_count],
                'Indirect_Strength_Mean': [strength_sum / paths_count]
            }),
                                                       ignore_index=True)
    indirect_info_df.to_csv(BaseConfig.OUT_PATH + 'SelfNetworkGrouply//' +
                            'Edges_of_linktype_' + p_name +
                            '_indirect_strength.csv')


def self_calculate_most_edges_in_linktype(p_linktypenet):
    vars_weightiest_var_ap = {}
    for target_var in VARS_LIST:
        causal_strengths = {}
        for var in VARS_LIST:
            if var == target_var:
                continue
            if not var in p_linktypenet.nodes():
                continue
            try:
                short_path = nx.shortest_path(p_linktypenet,
                                              source=var,
                                              target=target_var)
            except:
                continue
            strength_sum = 0
            paths_count = 0
            sum = 0
            weights = 0
            for i, n_list in enumerate(short_path):
                if i == len(short_path) - 1:
                    break
                props = p_linktypenet[short_path[i]][short_path[i + 1]]
                weights = weights + props['weight']
                sum = sum + weights
            a_path_strength = sum / (len(short_path) - 1)
            strength_sum = strength_sum + a_path_strength
            paths_count = paths_count + 1
            if paths_count == 0:
                continue
            causal_strengths[var] = strength_sum / paths_count
        causal_strengths_abs = dict(
            map(lambda x, y: [x, abs(y)], causal_strengths.keys(),
                causal_strengths.values()))
        causal_strengths_abs_sorted = sorted(causal_strengths_abs.items(),
                                             key=lambda x: x[1],
                                             reverse=True)
        vars_weightiest_var_ap[target_var] = causal_strengths_abs_sorted
    return vars_weightiest_var_ap


def self_draw_shortest_path_causal_agents_by_var(p_alllink_df,
                                                 p_vars_weightiest_var_ap,
                                                 p_draw_var):
    fig = plt.figure(figsize=(28, 12), dpi=500)
    # [左, 下, 宽, 高] 规定的矩形区域 （全部是0~1之间的数，表示比例）
    # rect_ax = [0.1, 0.1, 0.8, 0.78]
    # rectCB = [0, 0.75, 1, 0.2]
    geoagent_shp_path = BaseConfig.GEO_AGENT_PATH + 'GA_WGS84_YR.shp'
    targetPro = ccrs.PlateCarree()
    for var_est in range(0, 4):
        strength_df = self_calculate_shortest_path_causal_strength(
            p_alllink_df, p_vars_weightiest_var_ap[var_est][0], p_draw_var)
        # draw map
        ax = fig.add_subplot(2, 2, var_est + 1, projection=targetPro)
        # ax = plt.axes(projection=targetPro)
        # ax.set_global()
        # ax.stock_img()
        ax.add_feature(cfeature.OCEAN.with_scale('50m'))
        ax.add_feature(cfeature.LAND.with_scale('50m'))
        ax.add_feature(cfeature.RIVERS.with_scale('50m'))
        ax.add_feature(cfeature.LAKES.with_scale('50m'))
        ax.set_extent([95, 120, 31.5, 42])
        ax.set_title('Shortest Path Causal Strength from ' +
                     p_vars_weightiest_var_ap[var_est][0] + ' to ' +
                     p_draw_var)

        feature_class = Reader(geoagent_shp_path).records()
        for feature in feature_class:
            ga_id = int(feature.attributes['GA_ID'])
            polygon_geo = ShapelyFeature([feature.geometry],
                                         ccrs.PlateCarree())
            hatch_str = ''
            weightiest_var = ''
            strength_sign = ''
            strength_min = strength_df['Strength'].min()
            strength_max = strength_df['Strength'].max()
            if strength_df.loc[ga_id]['Strength'] == 0:
                color = '#ffffff'
            else:
                if strength_max == strength_min:
                    color = '#008000'
                else:
                    color = cm.Greens(
                        (strength_df.loc[ga_id]['Strength'] - strength_min) /
                        (strength_max - strength_min))
            # if ga_id in p_agents_weightiest_var_dict:
            ax.add_feature(polygon_geo,
                           linewidth=0.4,
                           facecolor=color,
                           edgecolor='#4D5459',
                           alpha=0.8)

    plt.savefig(BaseConfig.OUT_PATH +
                'SelfNetworkFigs//Self_draw_shortest_path_causal_to_' +
                p_draw_var + '.pdf')
    # axCB = plt.axes(rectCB)
    # axCB.spines['top'].set_visible(False)
    # axCB.spines['right'].set_visible(False)
    # axCB.spines['bottom'].set_visible(False)
    # axCB.spines['left'].set_visible(False)
    # axCB.set_xticks([])
    # axCB.set_xticks([])
    # cMap = ListedColormap(VAR_COLOR_DICT.values())
    # cNorm = BoundaryNorm(np.arange(1 + cMap.N), cMap.N)
    # cb = fig.colorbar(plt.cm.ScalarMappable(norm=cNorm, cmap=cMap),
    #                   ax=axCB,
    #                   orientation='horizontal',
    #                   aspect=34)
    # cb.set_ticks(np.arange(0.5, 1.5 + cMap.N))
    # cb.set_ticklabels(list(VAR_COLOR_DICT.keys()))
    # cb.set_ticklabels(
    #     list({x: VAR_LABEL_DICT[x]
    #           for x in list(VAR_COLOR_DICT.keys())}.values()))
    # for l in cb.ax.xaxis.get_ticklabels():
    #     l.set_family('Times New Roman')
    #     l.set_size(12)


def self_calculate_shortest_path_causal_strength(p_filtered_df, p_est_var,
                                                 p_draw_var):
    centroid_data = pd.read_csv(BaseConfig.GEO_AGENT_PATH + 'GA_Centroid.csv')
    weightiest_info_df = pd.DataFrame(
        columns=['GA_ID', 'Weightiest_var', 'Strength'])
    for agent in list(centroid_data['GA_ID']):
        weightiest_info_df.loc[agent, 'GA_ID'] = agent
        if agent in list(p_filtered_df['Source'].unique()):
            agent_self_net = build_net_by_links_df(
                p_filtered_df[(p_filtered_df['Source'] == agent)
                              & (p_filtered_df['Target'] == agent)].copy())
            if not str(agent) + '_' + p_draw_var in agent_self_net.nodes():
                weightiest_info_df.loc[agent, 'Strength'] = 0
                continue
            try:
                short_path = nx.shortest_path(
                    agent_self_net,
                    source=str(agent) + '_' + p_est_var,
                    target=str(agent) + '_' + p_draw_var)
            except:
                short_path = {}
            # del short_paths[str(agent) + '_' + p_draw_var]
            if short_path == {}:
                weightiest_info_df.loc[agent, 'Strength'] = 0
                continue
            weightiest_info_df.loc[agent, 'Weightiest_var'] = p_est_var
            weightiest_info_df.loc[
                agent, 'Strength'] = self_calculate_sum_weight_of_path(
                    agent_self_net, short_path)
    weightiest_info_df.to_csv(BaseConfig.OUT_PATH + 'SelfNetworkFigs//' +
                              'Info_' + p_est_var + '_to_' + p_draw_var +
                              '_in_agents.csv')
    return weightiest_info_df


def self_draw_agents_weightiest_var(p_agents_weightiest_var_dict, p_fig_name):
    # draw map
    fig = plt.figure(figsize=(20, 15), dpi=500)
    # [左, 下, 宽, 高] 规定的矩形区域 （全部是0~1之间的数，表示比例）
    rect_ax = [0.1, 0.1, 0.8, 0.78]
    rectCB = [0, 0.75, 1, 0.2]

    geoagent_shp_path = BaseConfig.GEO_AGENT_PATH + 'GA_WGS84.shp'
    targetPro = ccrs.PlateCarree()
    # ax = fig.add_subplot(1, 2, 1, projection=targetPro)
    ax = plt.axes(rect_ax, projection=targetPro)
    # ax.set_global()
    # ax.stock_img()
    ax.add_feature(cfeature.OCEAN.with_scale('50m'))
    ax.add_feature(cfeature.LAND.with_scale('50m'))
    ax.add_feature(cfeature.RIVERS.with_scale('50m'))
    ax.add_feature(cfeature.LAKES.with_scale('50m'))
    ax.set_extent([95, 120, 31.5, 42])

    feature_class = Reader(geoagent_shp_path).records()
    for feature in feature_class:
        ga_id = int(feature.attributes['GA_ID'])
        polygon_geo = ShapelyFeature([feature.geometry], ccrs.PlateCarree())
        hatch_str = ''
        weightiest_var = ''
        strength_sign = ''
        if ga_id in p_agents_weightiest_var_dict:
            if p_agents_weightiest_var_dict[ga_id] == 'Uncertain':
                face_color = '#CCCCCC'
                edge_color = 'k'
                weightiest_var = 'Uncertain'
                strength_sign = 'Uncertain'
                hatch_str = '*'
            else:
                weightiest_var = str(
                    p_agents_weightiest_var_dict[ga_id]).split(':')[0]
                strength_sign = str(
                    p_agents_weightiest_var_dict[ga_id]).split(':')[1]
                face_color = VAR_COLOR_DICT[weightiest_var]
                if strength_sign == '+':
                    edge_color = '#454545'
                    hatch_str = '..'
                else:
                    edge_color = '#454545'
                    hatch_str = '///'
        else:
            face_color = '#CCCCCC'
            edge_color = 'k'
            weightiest_var = 'Uncertain'
            strength_sign = 'Uncertain'
        ax.add_feature(polygon_geo,
                       linewidth=0.4,
                       facecolor=face_color,
                       edgecolor=edge_color,
                       alpha=1,
                       hatch=hatch_str)

    axCB = plt.axes(rectCB)
    axCB.spines['top'].set_visible(False)
    axCB.spines['right'].set_visible(False)
    axCB.spines['bottom'].set_visible(False)
    axCB.spines['left'].set_visible(False)
    axCB.set_xticks([])
    axCB.set_xticks([])
    cMap = ListedColormap(VAR_COLOR_DICT.values())
    cNorm = BoundaryNorm(np.arange(1 + cMap.N), cMap.N)
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=cNorm, cmap=cMap),
                      ax=axCB,
                      orientation='horizontal',
                      aspect=34)
    cb.set_ticks(np.arange(0.5, 1.5 + cMap.N))
    cb.set_ticklabels(list(VAR_COLOR_DICT.keys()))
    cb.set_ticklabels(
        list({x: VAR_LABEL_DICT[x]
              for x in list(VAR_COLOR_DICT.keys())}.values()))
    for l in cb.ax.xaxis.get_ticklabels():
        l.set_family('Times New Roman')
        l.set_size(12)

    plt.savefig(BaseConfig.OUT_PATH + 'SelfNetworkFigs//' + p_fig_name +
                '.pdf')


def self_draw_net_for_agent(p_self_net, p_agent_name):
    """
    draw self net for a agent
    """
    pos_old = nx.circular_layout(p_self_net)
    pos_new_keys = []
    for var in VARS_LIST:
        if str(p_agent_name) + '_' + var in pos_old.keys():
            pos_new_keys.append(str(p_agent_name) + '_' + var)
    pos = dict(map(lambda x, y: [x, y], pos_new_keys, pos_old.values()))
    fig = plt.figure(figsize=(5, 5), dpi=300)
    # ax = fig.add_subplot(111, frame_on=False)
    ax = plt.gca()
    x_values, y_values = zip(*pos.values())
    x_max = max(x_values)
    x_min = min(x_values)
    x_margin = (x_max - x_min) * 0.25
    plt.xlim(x_min - x_margin, x_max + x_margin)
    y_max = max(y_values)
    y_min = min(y_values)
    y_margin = (y_max - y_min) * 0.25
    plt.ylim(y_min - y_margin, y_max + y_margin)

    for n, d in p_self_net.nodes(data=True):
        var_name = str(n).split('_')[1]
        var_label = VAR_LABEL_DICT[var_name]
        p_self_net.nodes[n]['label'] = var_label
        color_edge = VAR_COLOR_DICT[var_name]
        if VARS_TIME_SCALE_DICT[var_name] == 'yearly':
            color_edge = '#454545'
        c = Ellipse(
            pos[n],
            width=0.2,
            height=0.2,
            clip_on=False,
            facecolor=VAR_COLOR_DICT[var_name],
            edgecolor=color_edge,
            zorder=0,
        )
        ax.add_patch(c)
        p_self_net.nodes[n]["patch"] = c

    cm = plt.cm.RdBu_r
    cNorm = matplotlib.colors.Normalize(vmin=-1, vmax=1)
    for (u, v, d) in p_self_net.edges(data=True):
        rad = str(0)
        width = 2.5
        zorder_value = -1
        if d['timelag'] == 1:
            rad = str(-0.2)
            width = 2
            zorder_value = -2
        elif d['timelag'] >= 2:
            rad = str(-0.3)
            width = 1.5
            zorder_value = -3
        e_p = FancyArrowPatch(
            pos[u],
            pos[v],
            arrowstyle='->,head_length=0.4, head_width=0.2',
            connectionstyle='arc3,rad=' + rad,
            mutation_scale=10,
            lw=width,
            alpha=0.9,
            linestyle='-',
            color=cm(cNorm(d['weight'])),
            clip_on=False,
            patchA=p_self_net.nodes[u]["patch"],
            patchB=p_self_net.nodes[v]["patch"],
            shrinkA=0,
            shrinkB=0,
            zorder=zorder_value,
        )
        ax.add_artist(e_p)

        # Attach labels of lags
        if d['timelag'] != 0:
            trans = None
            path = e_p.get_path()
            verts = path.to_polygons(trans)[0]
            if len(verts) > 2:
                label_vert = verts[1, :]
                string = str(d['timelag'])
                txt = ax.text(
                    label_vert[0],
                    label_vert[1],
                    string,
                    fontsize=4,
                    verticalalignment="center",
                    horizontalalignment="center",
                    color="w",
                    alpha=0.8,
                    zorder=2,
                )
                txt.set_path_effects(
                    [PathEffects.withStroke(linewidth=0.5, foreground="k")])

    # nx.draw_networkx_edges(
    #     p_self_net,
    #     pos=pos,
    #     width=4,
    #     # width=[float(d['weight']) for (u, v, d) in p_self_net.edges(data=True)],
    #     alpha=0.8,
    #     style='solid',
    #     # edge_color='k',
    #     # edge_color=[
    #     #     VAR_COLOR_DICT[p_self_net.nodes[u]['var_name']]
    #     #     for (u, v, d) in p_self_net.edges(data=True)
    #     # ],
    #     edge_color=[
    #         float(d['weight']) for (u, v, d) in p_self_net.edges(data=True)
    #     ],
    #     edge_cmap=plt.cm.RdBu_r,
    #     edge_vmin=-1,
    #     edge_vmax=1,
    #     arrows=True,
    #     arrowstyle='->',
    #     arrowsize=4,
    #     connectionstyle="arc3,rad=0.2")

    # nx.draw_networkx_nodes(p_self_net,
    #                        pos,
    #                        node_size=600,
    #                        node_color=[
    #                            VAR_COLOR_DICT[str(n).split('_')[1]]
    #                            for n, d in p_self_net.nodes(data=True)
    #                        ],
    #                        label=[
    #                            p_self_net.nodes[n]['label']
    #                            for n, d in p_self_net.nodes(data=True)
    #                        ])

    nx.draw_networkx_labels(p_self_net,
                            pos,
                            labels={
                                n: p_self_net.nodes[n]['label']
                                for n, d in p_self_net.nodes(data=True)
                            },
                            font_size=5,
                            font_color='k')
    plt.savefig(BaseConfig.OUT_PATH + 'SelfNetworkFigs//' + str(p_agent_name) +
                '_self_network.pdf',
                bbox_inches='tight')


def self_calculate_shortest_path_causal(p_filtered_df, p_target_var):
    centroid_data = pd.read_csv(BaseConfig.GEO_AGENT_PATH + 'GA_Centroid.csv')
    agents_weightiest_var_sp = {}
    weightiest_info_df = pd.DataFrame(
        columns=['GA_ID', 'Weightiest_var', 'Strength_sign'])
    for agent in list(centroid_data['GA_ID']):
        if agent in list(p_filtered_df['Source'].unique()):
            agent_self_net = build_net_by_links_df(
                p_filtered_df[(p_filtered_df['Source'] == agent)
                              & (p_filtered_df['Target'] == agent)].copy())
            if not str(agent) + '_' + p_target_var in agent_self_net.nodes():
                agents_weightiest_var_sp[agent] = 'Uncertain'
                continue
            short_paths = nx.single_target_shortest_path(
                agent_self_net,
                str(agent) + '_' + p_target_var)
            del short_paths[str(agent) + '_' + p_target_var]
            if short_paths == {}:
                agents_weightiest_var_sp[agent] = 'Uncertain'
                continue
            causal_strengths = {}
            for source_n in short_paths:
                causal_strengths[source_n] = self_calculate_sum_weight_of_path(
                    agent_self_net, short_paths[source_n])
                length_filed = 'Shortest_path_from_' + str(source_n).split(
                    '_')[1] + '_length'
                strength_field = 'Shortest_path_from_' + str(source_n).split(
                    '_')[1] + '_strength'
                if length_filed not in weightiest_info_df.columns:
                    weightiest_info_df[length_filed] = None
                    weightiest_info_df.loc[agent, length_filed] = len(
                        short_paths[source_n])
                else:
                    weightiest_info_df.loc[agent, length_filed] = len(
                        short_paths[source_n])

                if strength_field not in weightiest_info_df.columns:
                    weightiest_info_df[strength_field] = None
                    weightiest_info_df.loc[
                        agent, strength_field] = causal_strengths[source_n]
                else:
                    weightiest_info_df.loc[
                        agent, strength_field] = causal_strengths[source_n]
            causal_strengths_abs = dict(
                map(lambda x, y: [x, abs(y)], causal_strengths.keys(),
                    causal_strengths.values()))
            causal_strengths_abs_sorted = sorted(causal_strengths_abs.items(),
                                                 key=lambda x: x[1],
                                                 reverse=True)
            weightiest_var = str(
                causal_strengths_abs_sorted[0][0]).split('_')[1]
            sign = '+'
            if np.sign(
                    causal_strengths[causal_strengths_abs_sorted[0][0]]) < 0:
                sign = '-'
            agents_weightiest_var_sp[agent] = weightiest_var + ':' + str(sign)

            weightiest_info_df.loc[agent, 'Weightiest_var'] = weightiest_var
            weightiest_info_df.loc[agent, 'Strength_sign'] = str(sign)

    weightiest_info_df.to_csv(BaseConfig.OUT_PATH + 'SelfNetworkFigs//' +
                              'Agents_weightiest_var_sp_to_LAI.csv')
    return agents_weightiest_var_sp


def self_calculate_shortest_path_causal_pos(p_filtered_df, p_target_var):
    centroid_data = pd.read_csv(BaseConfig.GEO_AGENT_PATH + 'GA_Centroid.csv')
    agents_weightiest_var_sp_pos = {}
    weightiest_info_df = pd.DataFrame(
        columns=['GA_ID', 'Weightiest_var', 'Strength_sign'])

    for agent in list(centroid_data['GA_ID']):
        # weightiest_info_df = weightiest_info_df.append(
        #     pd.DataFrame({'GA_ID': agent}, index=[agent]))
        if agent in list(p_filtered_df['Source'].unique()):
            agent_self_net = build_net_by_links_df(
                p_filtered_df[(p_filtered_df['Source'] == agent)
                              & (p_filtered_df['Target'] == agent)].copy())

            if not str(agent) + '_' + p_target_var in agent_self_net.nodes():
                agents_weightiest_var_sp_pos[agent] = 'Uncertain'
                continue
            short_paths = nx.single_target_shortest_path(
                agent_self_net,
                str(agent) + '_' + p_target_var)
            del short_paths[str(agent) + '_' + p_target_var]
            if short_paths == {}:
                agents_weightiest_var_sp_pos[agent] = 'Uncertain'
                continue
            causal_strengths = {}
            for source_n in short_paths:
                causal_strengths[source_n] = self_calculate_sum_weight_of_path(
                    agent_self_net, short_paths[source_n])
                length_filed = 'Shortest_path_from_' + str(source_n).split(
                    '_')[1] + '_length'
                strength_field = 'Shortest_path_from_' + str(source_n).split(
                    '_')[1] + '_strength'
                if length_filed not in weightiest_info_df.columns:
                    weightiest_info_df[length_filed] = None
                    weightiest_info_df.loc[agent, length_filed] = len(
                        short_paths[source_n])
                else:
                    weightiest_info_df.loc[agent, length_filed] = len(
                        short_paths[source_n])

                if strength_field not in weightiest_info_df.columns:
                    weightiest_info_df[strength_field] = None
                    weightiest_info_df.loc[
                        agent, strength_field] = causal_strengths[source_n]
                else:
                    weightiest_info_df.loc[
                        agent, strength_field] = causal_strengths[source_n]

            causal_strengths_abs_sorted = sorted(causal_strengths.items(),
                                                 key=lambda x: x[1],
                                                 reverse=True)
            weightiest_var = str(
                causal_strengths_abs_sorted[0][0]).split('_')[1]
            sign = '+'
            if np.sign(
                    causal_strengths[causal_strengths_abs_sorted[0][0]]) < 0:
                sign = '-'
            agents_weightiest_var_sp_pos[agent] = weightiest_var + ':' + str(
                sign)

            weightiest_info_df.loc[agent, 'Weightiest_var'] = weightiest_var
            weightiest_info_df.loc[agent, 'Strength_sign'] = str(sign)

    weightiest_info_df.to_csv(BaseConfig.OUT_PATH + 'SelfNetworkFigs//' +
                              'Agents_weightiest_var_sp_pos_to_LAI.csv')
    return agents_weightiest_var_sp_pos


def self_calculate_all_path_causal(p_filtered_df, p_target_var):
    centroid_data = pd.read_csv(BaseConfig.GEO_AGENT_PATH + 'GA_Centroid.csv')
    agents_weightiest_var_ap = {}
    weightiest_info_df = pd.DataFrame(
        columns=['GA_ID', 'Weightiest_var', 'Strength_sign'])
    for agent in list(centroid_data['GA_ID']):
        if agent in list(p_filtered_df['Source'].unique()):
            agent_self_net = build_net_by_links_df(
                p_filtered_df[(p_filtered_df['Source'] == agent)
                              & (p_filtered_df['Target'] == agent)].copy())
            causal_strengths = {}
            for var in VARS_LIST:
                if var == p_target_var:
                    continue
                source_n = str(agent) + '_' + var
                if not source_n in agent_self_net.nodes():
                    continue
                all_paths = nx.all_simple_paths(
                    agent_self_net, source_n,
                    str(agent) + '_' + p_target_var, 5)
                strength_sum = 0
                paths_count = 0
                for a_path in all_paths:
                    a_path_strength = self_calculate_sum_weight_of_path(
                        agent_self_net, a_path)
                    strength_sum = strength_sum + a_path_strength
                    paths_count = paths_count + 1
                if paths_count == 0:
                    continue
                causal_strengths[source_n] = strength_sum / paths_count

                length_filed = 'All_path_from_' + str(source_n).split(
                    '_')[1] + '_count'
                strength_field = 'All_path_from_' + str(source_n).split(
                    '_')[1] + '_strength'
                if length_filed not in weightiest_info_df.columns:
                    weightiest_info_df[length_filed] = None
                    weightiest_info_df.loc[agent, length_filed] = paths_count
                else:
                    weightiest_info_df.loc[agent, length_filed] = paths_count

                if strength_field not in weightiest_info_df.columns:
                    weightiest_info_df[strength_field] = None
                    weightiest_info_df.loc[
                        agent, strength_field] = causal_strengths[source_n]
                else:
                    weightiest_info_df.loc[
                        agent, strength_field] = causal_strengths[source_n]

            if causal_strengths == {}:
                agents_weightiest_var_ap[agent] = 'Uncertain'
                continue
            causal_strengths_abs = dict(
                map(lambda x, y: [x, abs(y)], causal_strengths.keys(),
                    causal_strengths.values()))
            causal_strengths_abs_sorted = sorted(causal_strengths_abs.items(),
                                                 key=lambda x: x[1],
                                                 reverse=True)
            weightiest_var = str(
                causal_strengths_abs_sorted[0][0]).split('_')[1]
            sign = '+'
            if np.sign(
                    causal_strengths[causal_strengths_abs_sorted[0][0]]) < 0:
                sign = '-'
            agents_weightiest_var_ap[agent] = weightiest_var + ':' + str(sign)

            weightiest_info_df.loc[agent, 'Weightiest_var'] = weightiest_var
            weightiest_info_df.loc[agent, 'Strength_sign'] = str(sign)

    weightiest_info_df.to_csv(BaseConfig.OUT_PATH + 'SelfNetworkFigs//' +
                              'Agents_weightiest_var_ap_to_LAI.csv')
    return agents_weightiest_var_ap


def self_calculate_all_path_causal_pos(p_filtered_df, p_target_var):
    centroid_data = pd.read_csv(BaseConfig.GEO_AGENT_PATH + 'GA_Centroid.csv')
    agents_weightiest_var_ap_pos = {}
    weightiest_info_df = pd.DataFrame(
        columns=['GA_ID', 'Weightiest_var', 'Strength_sign'])
    for agent in list(centroid_data['GA_ID']):
        if agent in list(p_filtered_df['Source'].unique()):
            agent_self_net = build_net_by_links_df(
                p_filtered_df[(p_filtered_df['Source'] == agent)
                              & (p_filtered_df['Target'] == agent)].copy())
            causal_strengths = {}
            for var in VARS_LIST:
                if var == p_target_var:
                    continue
                source_n = str(agent) + '_' + var
                if not source_n in agent_self_net.nodes():
                    continue
                all_paths = nx.all_simple_paths(
                    agent_self_net, source_n,
                    str(agent) + '_' + p_target_var, 5)
                strength_sum = 0
                paths_count = 0
                for a_path in all_paths:
                    a_path_strength = self_calculate_sum_weight_of_path(
                        agent_self_net, a_path)
                    strength_sum = strength_sum + a_path_strength
                    paths_count = paths_count + 1
                if paths_count == 0:
                    continue
                causal_strengths[source_n] = strength_sum / paths_count

                length_filed = 'All_path_from_' + str(source_n).split(
                    '_')[1] + '_count'
                strength_field = 'All_path_from_' + str(source_n).split(
                    '_')[1] + '_strength'
                if length_filed not in weightiest_info_df.columns:
                    weightiest_info_df[length_filed] = None
                    weightiest_info_df.loc[agent, length_filed] = paths_count
                else:
                    weightiest_info_df.loc[agent, length_filed] = paths_count

                if strength_field not in weightiest_info_df.columns:
                    weightiest_info_df[strength_field] = None
                    weightiest_info_df.loc[
                        agent, strength_field] = causal_strengths[source_n]
                else:
                    weightiest_info_df.loc[
                        agent, strength_field] = causal_strengths[source_n]

            if causal_strengths == {}:
                agents_weightiest_var_ap_pos[agent] = 'Uncertain'
                continue
            causal_strengths_abs_sorted = sorted(causal_strengths.items(),
                                                 key=lambda x: x[1],
                                                 reverse=True)
            weightiest_var = str(
                causal_strengths_abs_sorted[0][0]).split('_')[1]
            sign = '+'
            if np.sign(
                    causal_strengths[causal_strengths_abs_sorted[0][0]]) < 0:
                sign = '-'
            agents_weightiest_var_ap_pos[agent] = weightiest_var + ':' + str(
                sign)

            weightiest_info_df.loc[agent, 'Weightiest_var'] = weightiest_var
            weightiest_info_df.loc[agent, 'Strength_sign'] = str(sign)

    weightiest_info_df.to_csv(BaseConfig.OUT_PATH + 'SelfNetworkFigs//' +
                              'Agents_weightiest_var_ap_pos_to_LAI.csv')
    return agents_weightiest_var_ap_pos


def self_calculate_sum_weight_of_path(p_net, p_nodes_list):
    sum = 0
    for i, n_list in enumerate(p_nodes_list):
        if i == len(p_nodes_list) - 1:
            break
        props = p_net[p_nodes_list[i]][p_nodes_list[i + 1]]
        weights = 0
        for prop in props.values():
            weights = weights + prop['weight']
        sum = sum + weights
    if len(p_nodes_list) == 1:
        return 0
    else:
        return sum / (len(p_nodes_list) - 1)


def inner_analyze_nets():
    """
    output inner net info
    """
    # # draw inner net centrality info in GIS
    # new_pool = multiprocessing.Pool()
    # for var in VARS_LIST:
    #     p = new_pool.apply_async(draw_inner_centrality_info, args=(var, ))
    #     print('draw_inner_info_in_GIS:', var)
    # new_pool.close()
    # new_pool.join()

    # draw inner net communities info in GIS
    new_pool = multiprocessing.Pool()
    for var in VARS_LIST:
        p = new_pool.apply_async(inner_draw_communities_map_by_var,
                                 args=(var, ))
        print('draw_inner_info_in_GIS:', var)
    new_pool.close()
    new_pool.join()

    # set multiprocessing to get info
    new_pool = multiprocessing.Pool()
    jobs = []
    for var in VARS_LIST:
        p = new_pool.apply_async(inner_get_info_by_var, args=(var, ))
        jobs.append(p)
        print('get_inner_info_by_var:', var)
    new_pool.close()
    new_pool.join()
    kinds_links = []
    for job in jobs:
        kinds_links.append(job.get())
    nets_info = pd.concat(kinds_links, ignore_index=True)
    nets_info.to_csv(BaseConfig.OUT_PATH + 'All_Inner_Nets_Info.csv')


def inner_get_info_by_var(p_var):
    """
    get inner based info by var 
    """
    net_info = pd.DataFrame()
    var_inner_df = pd.read_csv(BaseConfig.OUT_PATH + 'InnerNetworkCSV//' +
                               p_var + '_filtered.csv')
    var_inner_net = build_net_by_links_df(var_inner_df)

    net_info = net_info.append(
        pd.DataFrame(
            {
                'Name':
                'Inner Net of ' + p_var,
                'Number of nodes':
                nx.number_of_nodes(var_inner_net),
                'Number of edges':
                nx.number_of_edges(var_inner_net),
                'Average Node Degree':
                sum(d for n, d in var_inner_net.degree()) /
                float(nx.number_of_nodes(var_inner_net)),
                'Density':
                nx.density(var_inner_net),
                'Transitivity':
                nx.transitivity(nx.DiGraph(var_inner_net)),
                'Average Node Connectivity':
                nx.average_node_connectivity(var_inner_net),
            },
            index=[0]))
    return net_info


def inner_draw_communities_map_by_var(p_var):
    """
    draw inner communities info on GIS
    """
    var_inner_df = pd.read_csv(BaseConfig.OUT_PATH + 'InnerNetworkCSV//' +
                               p_var + '_filtered.csv')
    var_inner_net = build_net_by_links_df(var_inner_df)
    # community_list=community.girvan_newman(var_inner_net)
    # comp_tuple=tuple(sorted(c) for c in next(community_list))
    # print(comp_tuple)
    # print(len(comp_tuple))
    community_list = community.asyn_fluidc(nx.MultiGraph(var_inner_net), 6)
    com_group_number = {}
    for c_index, com in enumerate(list(community_list)):
        for n in com:
            com_group_number[n] = c_index
    cmap = plt.get_cmap('tab10')

    fig = plt.figure(figsize=(20, 12), dpi=500)
    geoagent_shp_path = BaseConfig.GEO_AGENT_PATH + 'GA_WGS84.shp'
    targetPro = ccrs.PlateCarree()
    ax = fig.add_subplot(1, 1, 1, projection=targetPro)
    # ax.set_global()
    # ax.stock_img()
    ax.add_feature(cfeature.OCEAN.with_scale('50m'))
    ax.add_feature(cfeature.LAND.with_scale('50m'))
    ax.add_feature(cfeature.RIVERS.with_scale('50m'))
    ax.add_feature(cfeature.LAKES.with_scale('50m'))
    ax.set_extent([95, 120, 31.5, 42])

    for feature in Reader(geoagent_shp_path).records():
        ga_id = feature.attributes['GA_ID']
        polygon_geo = ShapelyFeature(feature.geometry, ccrs.PlateCarree())
        try:
            face_color = cmap(com_group_number[str(ga_id) + '_' + p_var])
        except:
            face_color = '#737373'
        ax.add_feature(polygon_geo,
                       linewidth=0.4,
                       facecolor=face_color,
                       edgecolor='#4D5459',
                       alpha=0.8)
    plt.savefig(BaseConfig.OUT_PATH + 'InnerNetCommunitiesMap//' + p_var +
                '_Communities_info_Map.pdf')


def inner_draw_centrality_map_by_var(p_var):
    """
    draw inner centrality info on GIS
    """
    var_inner_df = pd.read_csv(BaseConfig.OUT_PATH + 'InnerNetworkCSV//' +
                               p_var + '_filtered.csv')
    var_inner_net = build_net_by_links_df(var_inner_df)

    # draw degree map
    fig = plt.figure(figsize=(20, 12), dpi=500)
    geoagent_shp_path = BaseConfig.GEO_AGENT_PATH + 'GA_WGS84.shp'
    targetPro = ccrs.PlateCarree()
    ax = fig.add_subplot(2, 2, 1, projection=targetPro)
    # ax.set_global()
    # ax.stock_img()
    ax.add_feature(cfeature.OCEAN.with_scale('50m'))
    ax.add_feature(cfeature.LAND.with_scale('50m'))
    ax.add_feature(cfeature.RIVERS.with_scale('50m'))
    ax.add_feature(cfeature.LAKES.with_scale('50m'))
    ax.set_extent([95, 120, 31.5, 42])
    cmap = plt.get_cmap('Reds')
    norm = matplotlib.colors.Normalize(
        vmin=min(d for n, d in var_inner_net.degree()),
        vmax=max(d for n, d in var_inner_net.degree()))
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    cb = fig.colorbar(sm, ax=ax, orientation='horizontal')
    for l in cb.ax.xaxis.get_ticklabels():
        l.set_family('Times New Roman')
        l.set_size(14)
    cb.set_label('Degree', fontsize=14, fontfamily='Times New Roman')
    for feature in Reader(geoagent_shp_path).records():
        ga_id = feature.attributes['GA_ID']
        polygon_geo = ShapelyFeature(feature.geometry, ccrs.PlateCarree())
        try:
            face_color = sm.to_rgba(var_inner_net.degree[str(ga_id) + '_' +
                                                         p_var])
        except:
            face_color = '#737373'
        ax.add_feature(polygon_geo,
                       linewidth=0.4,
                       facecolor=face_color,
                       edgecolor='#4D5459',
                       alpha=0.8)

    # draw out_degree/indegree
    ax = fig.add_subplot(2, 2, 2, projection=targetPro)
    # ax.set_global()
    # ax.stock_img()
    ax.add_feature(cfeature.OCEAN.with_scale('50m'))
    ax.add_feature(cfeature.LAND.with_scale('50m'))
    ax.add_feature(cfeature.RIVERS.with_scale('50m'))
    ax.add_feature(cfeature.LAKES.with_scale('50m'))
    ax.set_extent([95, 120, 31.5, 42])
    cmap = plt.get_cmap('Reds')
    o_divide_i_values = []
    for n, d in var_inner_net.out_degree():
        try:
            o_divide_i_values.append(d / var_inner_net.in_degree(n))
        except:
            continue
    norm = matplotlib.colors.Normalize(vmin=min(o_divide_i_values),
                                       vmax=max(o_divide_i_values))
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    cb = fig.colorbar(sm, ax=ax, orientation='horizontal')
    for l in cb.ax.xaxis.get_ticklabels():
        l.set_family('Times New Roman')
        l.set_size(14)
    cb.set_label('out_degree/indegree',
                 fontsize=14,
                 fontfamily='Times New Roman')
    for feature in Reader(geoagent_shp_path).records():
        ga_id = feature.attributes['GA_ID']
        polygon_geo = ShapelyFeature(feature.geometry, ccrs.PlateCarree())
        try:
            face_color = sm.to_rgba(
                (var_inner_net.out_degree[str(ga_id) + '_' + p_var]) /
                (var_inner_net.in_degree[str(ga_id) + '_' + p_var]))
        except:
            face_color = '#737373'
        ax.add_feature(polygon_geo,
                       linewidth=0.4,
                       facecolor=face_color,
                       edgecolor='#4D5459',
                       alpha=0.8)

    # draw betweenness map
    ax = fig.add_subplot(2, 2, 3, projection=targetPro)
    # ax.set_global()
    # ax.stock_img()
    ax.add_feature(cfeature.OCEAN.with_scale('50m'))
    ax.add_feature(cfeature.LAND.with_scale('50m'))
    ax.add_feature(cfeature.RIVERS.with_scale('50m'))
    ax.add_feature(cfeature.LAKES.with_scale('50m'))
    ax.set_extent([95, 120, 31.5, 42])

    cmap = plt.get_cmap('Purples')
    norm = matplotlib.colors.Normalize(
        vmin=min(
            nx.betweenness_centrality(nx.DiGraph(var_inner_net)).values()),
        vmax=max(
            nx.betweenness_centrality(nx.DiGraph(var_inner_net)).values()))
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    cb = fig.colorbar(sm, ax=ax, orientation='horizontal')
    for l in cb.ax.xaxis.get_ticklabels():
        l.set_family('Times New Roman')
        l.set_size(14)
    cb.set_label('Betweenness', fontsize=14, fontfamily='Times New Roman')
    for feature in Reader(geoagent_shp_path).records():
        ga_id = feature.attributes['GA_ID']
        polygon_geo = ShapelyFeature(feature.geometry, ccrs.PlateCarree())
        try:
            face_color = sm.to_rgba(
                nx.betweenness_centrality(
                    nx.DiGraph(var_inner_net))[str(ga_id) + '_' + p_var])
        except:
            face_color = '#737373'
        ax.add_feature(polygon_geo,
                       linewidth=0.4,
                       facecolor=face_color,
                       edgecolor='#4D5459',
                       alpha=0.8)

    # draw Closeness map
    ax = fig.add_subplot(2, 2, 4, projection=targetPro)
    # ax.set_global()
    # ax.stock_img()
    ax.add_feature(cfeature.OCEAN.with_scale('50m'))
    ax.add_feature(cfeature.LAND.with_scale('50m'))
    ax.add_feature(cfeature.RIVERS.with_scale('50m'))
    ax.add_feature(cfeature.LAKES.with_scale('50m'))
    ax.set_extent([95, 120, 31.5, 42])

    cmap = plt.get_cmap('Blues')
    norm = matplotlib.colors.Normalize(
        vmin=min(nx.closeness_centrality(var_inner_net).values()),
        vmax=max(nx.closeness_centrality(var_inner_net).values()))
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    cb = fig.colorbar(sm, ax=ax, orientation='horizontal')
    for l in cb.ax.xaxis.get_ticklabels():
        l.set_family('Times New Roman')
        l.set_size(14)
    cb.set_label('Closeness', fontsize=14, fontfamily='Times New Roman')
    for feature in Reader(geoagent_shp_path).records():
        ga_id = feature.attributes['GA_ID']
        polygon_geo = ShapelyFeature(feature.geometry, ccrs.PlateCarree())
        try:
            face_color = sm.to_rgba(
                nx.closeness_centrality(var_inner_net)[str(ga_id) + '_' +
                                                       p_var])
        except:
            face_color = '#737373'
        ax.add_feature(polygon_geo,
                       linewidth=0.4,
                       facecolor=face_color,
                       edgecolor='#4D5459',
                       alpha=0.8)

    plt.savefig(BaseConfig.OUT_PATH + 'InnerNetCentralityMap//' + p_var +
                '_Centrality_info_Map.pdf')
    plt.close()


def build_self_links():
    """
    get self links for every GeoAgent
    """
    # out put the same columns
    # make a fake list include all times
    fake_months = []
    for iyear in np.arange(1955, 2030):
        for imonth in np.arange(1, 13):
            fake_months.append(str(iyear) + str(imonth).zfill(2))
    month_times_intersection = list(fake_months)
    year_times_intersection = list(map(str, np.arange(1955, 2030)))
    for var in VARS_LIST:
        # monthly data
        if VARS_TIME_SCALE_DICT[var] != 'yearly':
            var_data = pd.read_csv(BaseConfig.COUPLED_NET_DATA_PATH +
                                   BaseConfig.COUPLED_NET_DATA_HEAD + var +
                                   '_monthly' +
                                   BaseConfig.COUPLED_NET_DATA_TAIL,
                                   index_col='GA_ID')
            month_times_intersection = set(
                month_times_intersection).intersection(
                    set(list(var_data.columns.values)))
        # yearly data
        var_data = pd.read_csv(BaseConfig.COUPLED_NET_DATA_PATH +
                               BaseConfig.COUPLED_NET_DATA_HEAD + var +
                               '_yearly' + BaseConfig.COUPLED_NET_DATA_TAIL,
                               index_col='GA_ID')
        year_times_intersection = set(year_times_intersection).intersection(
            set(list(var_data.columns.values)))
    # sord them
    month_times_intersection = sorted(list(month_times_intersection))
    year_times_intersection = sorted(list(year_times_intersection))

    # set multiprocessing for every agent
    centroid_data = pd.read_csv(BaseConfig.GEO_AGENT_PATH + 'GA_Centroid.csv')
    new_pool = multiprocessing.Pool()
    jobs = []
    for agent in list(centroid_data['GA_ID']):
        p = new_pool.apply_async(build_self_links_individual,
                                 args=(agent, month_times_intersection,
                                       year_times_intersection))
        jobs.append(p)
    new_pool.close()
    new_pool.join()
    agents_links = []
    for job in jobs:
        agents_links.append(job.get())
    self_links = pd.concat(agents_links, ignore_index=True)
    self_links.to_csv(BaseConfig.OUT_PATH + 'SelfNetworkCSV//' +
                      'SelfNetworkAll' + '.csv')
    filtered_edges_df = stretch_filter_links(self_links)
    filtered_edges_df.to_csv(BaseConfig.OUT_PATH + 'SelfNetworkCSV//' +
                             'SelfNetworkAll_filtered' + '.csv')
    return self_links


def build_self_links_individual(p_agent, p_months_inter, p_years_inter):
    monthly_vars = []
    yearly_vars = []
    monthly_vars_data = []
    yearly_vars_data = []
    for var in VARS_LIST:
        # monthly data
        if VARS_TIME_SCALE_DICT[var] == 'monthly_yearly':
            var_data_monthly = pd.read_csv(BaseConfig.COUPLED_NET_DATA_PATH +
                                           BaseConfig.COUPLED_NET_DATA_HEAD +
                                           var + '_monthly' +
                                           BaseConfig.COUPLED_NET_DATA_TAIL,
                                           index_col='GA_ID')
            var_data_monthly.fillna(value=BaseConfig.BACKGROUND_VALUE,
                                    inplace=True)
            var_data_monthly = var_data_monthly[list(p_months_inter)]
            # filter all zero row
            same_counts_month = var_data_monthly.loc[p_agent].value_counts()
            v_count_m = var_data_monthly.loc[p_agent].count()
            if same_counts_month.values.max() < v_count_m - 24:
                monthly_vars_data.append(var_data_monthly.loc[p_agent])
                monthly_vars.append(var)
        # yearly data
        var_data = pd.read_csv(BaseConfig.COUPLED_NET_DATA_PATH +
                               BaseConfig.COUPLED_NET_DATA_HEAD + var +
                               '_yearly' + BaseConfig.COUPLED_NET_DATA_TAIL,
                               index_col='GA_ID')
        var_data.fillna(value=BaseConfig.BACKGROUND_VALUE, inplace=True)
        var_data = var_data[list(p_years_inter)]
        same_counts = var_data.loc[p_agent].value_counts()
        v_count_y = var_data.loc[p_agent].count()
        if same_counts.values.max() < v_count_y - 6:
            yearly_vars_data.append(var_data.loc[p_agent])
            yearly_vars.append(var)
    month_data_links = pd.DataFrame()
    year_data_links = pd.DataFrame()
    monthly_data = np.array(monthly_vars_data)
    # run month data drived links
    if monthly_data != []:
        month_data_links = build_link_pcmci_noself(monthly_data.T,
                                                   monthly_vars, '--', '--')
    yearly_data = np.array(yearly_vars_data)
    # run year data drived links
    if yearly_data != []:
        year_data_links = build_link_pcmci_noself(yearly_data.T, yearly_vars,
                                                  '--', '--')
    # move the yearly var links to the month data network
    only_yearly_vars = list(set(yearly_vars).difference(set(monthly_vars)))
    links_of_yearly_var = year_data_links.loc[
        (year_data_links['Source'].isin(only_yearly_vars))
        | (year_data_links['Target'].isin(only_yearly_vars))]

    agent_links = pd.concat([month_data_links, links_of_yearly_var],
                            ignore_index=True)
    # change the format of agent_links
    agent_links['VarSou'] = agent_links['Source']
    agent_links['VarTar'] = agent_links['Target']
    agent_links['Source'] = p_agent
    agent_links['Target'] = p_agent
    print('get_self_links_individual:', p_agent)
    print(time.strftime('%H:%M:%S', time.localtime(time.time())))
    return agent_links


def build_self_links_only_year_data():
    """
    get self links for every GeoAgent
    """
    # out put the same columns
    year_times_intersection = list(map(str, np.arange(1955, 2030)))
    for var in VARS_LIST:
        # yearly data
        var_data = pd.read_csv(BaseConfig.COUPLED_NET_DATA_PATH +
                               BaseConfig.COUPLED_NET_DATA_HEAD + var +
                               '_yearly' + BaseConfig.COUPLED_NET_DATA_TAIL,
                               index_col='GA_ID',
                               encoding='gbk')
        year_times_intersection = set(year_times_intersection).intersection(
            set(list(var_data.columns.values)))
    # sord them
    year_times_intersection = sorted(list(year_times_intersection))

    centroid_data = pd.read_csv(BaseConfig.GEO_AGENT_PATH + 'GA_Centroid.csv')
    agents_links = []
    for agent in list(centroid_data['GA_ID']):
        yearly_vars = []
        yearly_vars_data = []
        for var in VARS_LIST:
            # yearly data
            var_data = pd.read_csv(BaseConfig.COUPLED_NET_DATA_PATH +
                                   BaseConfig.COUPLED_NET_DATA_HEAD + var +
                                   '_yearly' +
                                   BaseConfig.COUPLED_NET_DATA_TAIL,
                                   index_col='GA_ID',
                                   encoding='gbk')
            var_data.fillna(value=BaseConfig.BACKGROUND_VALUE, inplace=True)
            var_data = var_data[list(year_times_intersection)]
            same_counts = var_data.loc[agent].value_counts()
            if same_counts.values.max() < 10:
                yearly_vars_data.append(var_data.loc[agent])
                yearly_vars.append(var)

        yearly_data = np.array(yearly_vars_data)
        # run year data drived links
        year_data_links = build_link_pcmci_noself(yearly_data.T, yearly_vars,
                                                  '--', '--')
        agent_links = year_data_links
        # change the format of agent_links
        agent_links['VarSou'] = agent_links['Source']
        agent_links['VarTar'] = agent_links['Target']
        agent_links['Source'] = agent
        agent_links['Target'] = agent
        agents_links.append(agent_links)
    self_links = pd.concat(agents_links, ignore_index=True)
    self_links.to_csv(BaseConfig.OUT_PATH + 'SelfNetworkCSV//' +
                      'SelfNetworkAll' + '.csv')
    filtered_edges_df = stretch_filter_links(self_links)
    filtered_edges_df.to_csv(BaseConfig.OUT_PATH + 'SelfNetworkCSV//' +
                             'SelfNetworkAll_filtered' + '.csv')
    return self_links


# 统一输入获得inner_links
# def build_inner_links(p_var_name):
#     """
#     get edgeLinks in same var Layer
#     @param p_var_name: var name string
#     @return:
#     """
#     data = None
#     if VARS_TIME_SCALE_DICT[p_var_name] == 'yearly':
#         data = pd.read_csv(BaseConfig.COUPLED_NET_DATA_PATH +
#                            BaseConfig.COUPLED_NET_DATA_HEAD + p_var_name +
#                            '_yearly' + BaseConfig.COUPLED_NET_DATA_TAIL)
#     else:
#         data = pd.read_csv(BaseConfig.COUPLED_NET_DATA_PATH +
#                            BaseConfig.COUPLED_NET_DATA_HEAD + p_var_name +
#                            '_monthly' + BaseConfig.COUPLED_NET_DATA_TAIL)
#     data.fillna(value=BaseConfig.BACKGROUND_VALUE, inplace=True)
#     data_values = data.values
#     id_data = data_values[..., 0].astype(np.int32)
#     var_names = list(map(str, id_data))
#     data_values = np.delete(data_values, 0, axis=1)
#     inner_links = build_link_pcmci_noself(data_values.T, var_names, p_var_name,
#                                           p_var_name)
#     return inner_links


# 逐条输入获得inner_links
def build_inner_links(p_var_name):
    """
    get edgeLinks in same var Layer
    @param p_var_name: var name string
    @return:
    """
    data = None
    if VARS_TIME_SCALE_DICT[p_var_name] == 'yearly':
        data = pd.read_csv(BaseConfig.COUPLED_NET_DATA_PATH +
                           BaseConfig.COUPLED_NET_DATA_HEAD + p_var_name +
                           '_yearly' + BaseConfig.COUPLED_NET_DATA_TAIL)
    else:
        data = pd.read_csv(BaseConfig.COUPLED_NET_DATA_PATH +
                           BaseConfig.COUPLED_NET_DATA_HEAD + p_var_name +
                           '_monthly' + BaseConfig.COUPLED_NET_DATA_TAIL)
    data.fillna(value=BaseConfig.BACKGROUND_VALUE, inplace=True)
    data_values = data.values
    id_data = data_values[..., 0].astype(np.int32)
    data_values = np.delete(data_values, 0, axis=1)
    [agent_num, times_num] = data_values.shape
    one_links = []
    for i_sou in np.arange(0, agent_num):
        this_sou = data_values[i_sou, ...]
        for i_tar in np.arange(0, agent_num - 1):
            data_2_row = np.array([this_sou, data_values[i_tar, ...]])
            try:
                one_link = build_link_pcmci_noself(
                    data_2_row.T, [id_data[i_sou], id_data[i_tar]], p_var_name,
                    p_var_name)
                one_links.append(one_link)
            except:
                continue
    inner_links = pd.concat(one_links, ignore_index=True)
    inner_links.to_csv(BaseConfig.OUT_PATH + 'InnerNetworkCSV//' + p_var_name +
                       '.csv')
    filtered_edges_df = stretch_filter_links(inner_links)
    filtered_edges_df.to_csv(BaseConfig.OUT_PATH + 'InnerNetworkCSV//' +
                             p_var_name + '_filtered.csv')
    return inner_links


def build_outer_links(p_var_sou, p_var_tar):
    """
    get edgeLinks in 2 diferent var Layer
    @param p_var_sou: source var name
    @param p_var_tar: target var name
    @return:
    """
    data_sou = None
    data_tar = None
    # monthly data
    if VARS_TIME_SCALE_DICT[p_var_sou] == VARS_TIME_SCALE_DICT[
            p_var_tar] == 'monthly_yearly':
        data_sou = pd.read_csv(BaseConfig.COUPLED_NET_DATA_PATH +
                               BaseConfig.COUPLED_NET_DATA_HEAD + p_var_sou +
                               '_monthly' + BaseConfig.COUPLED_NET_DATA_TAIL)
        data_sou.fillna(value=BaseConfig.BACKGROUND_VALUE, inplace=True)
        data_tar = pd.read_csv(BaseConfig.COUPLED_NET_DATA_PATH +
                               BaseConfig.COUPLED_NET_DATA_HEAD + p_var_tar +
                               '_monthly' + BaseConfig.COUPLED_NET_DATA_TAIL)
        data_tar.fillna(value=BaseConfig.BACKGROUND_VALUE, inplace=True)
        sou_times = list(data_sou.columns.values)
        tar_times = list(data_tar.columns.values)
        # out put the same columns
        same_times = []
        for i_sou_time in sou_times:
            for i_tar_time in tar_times:
                if i_sou_time == i_tar_time:
                    same_times.append(i_sou_time)
        # update the data to same scale and length
        data_sou = data_sou[same_times]
        data_tar = data_tar[same_times]
    # yearly data
    else:
        data_sou = pd.read_csv(BaseConfig.COUPLED_NET_DATA_PATH +
                               BaseConfig.COUPLED_NET_DATA_HEAD + p_var_sou +
                               '_yearly' + BaseConfig.COUPLED_NET_DATA_TAIL)
        data_sou.fillna(value=BaseConfig.BACKGROUND_VALUE)
        data_tar = pd.read_csv(BaseConfig.COUPLED_NET_DATA_PATH +
                               BaseConfig.COUPLED_NET_DATA_HEAD + p_var_tar +
                               '_yearly' + BaseConfig.COUPLED_NET_DATA_TAIL)
        data_tar.fillna(value=BaseConfig.BACKGROUND_VALUE, inplace=True)
        sou_times = list(data_sou.columns.values)
        tar_times = list(data_tar.columns.values)
        # out put the same columns
        same_times = []
        for i_sou_time in sou_times:
            for i_tar_time in tar_times:
                if i_sou_time == i_tar_time:
                    same_times.append(i_sou_time)
        # update the data to same scale and length
        data_sou = data_sou[same_times]
        data_tar = data_tar[same_times]

    data_sou_values = data_sou.values
    data_tar_values = data_tar.values
    id_sou = data_sou_values[..., 0].astype(np.int32)
    id_tar = data_tar_values[..., 0].astype(np.int32)
    sou_values = np.delete(data_sou_values, 0, axis=1)
    tar_values = np.delete(data_tar_values, 0, axis=1)
    sou_values_nor = sou_values
    tar_values_nor = tar_values
    # sou_values_nor = z_score_normalization(sou_values)
    # tar_values_nor = z_score_normalization(tar_values)
    [agent_num, times_num] = sou_values.shape
    one_links = []
    for i_sou in np.arange(0, agent_num):
        this_sou = sou_values_nor[i_sou, ...]
        for i_tar in np.arange(0, agent_num - 1):
            data_2_row = np.array([this_sou, tar_values_nor[i_tar, ...]])
            try:
                one_link = build_link_pcmci_noself(
                    data_2_row.T, [id_sou[i_sou], id_tar[i_tar]], p_var_sou,
                    p_var_tar)
                one_links.append(one_link)
            except:
                continue
    outer_links = pd.concat(one_links, ignore_index=True)
    return outer_links


# 弃用，使用igraph的构建方法
# def build_coupled_network_ig():
#     """
#     build coupled network by the edges df from csv by igraph
#     """
#     all_edges_df = pd.read_csv(BaseConfig.OUT_PATH +
#                                'Coupled_Network\\AllLinks.csv')
#     print(all_edges_df)
#     # build a net network
#     coupled_network = igraph.Graph(directed=True)
#     # add every vertex to the net
#     var_sou = all_edges_df['VarSou'].map(str)
#     var_tar = all_edges_df['VarTar'].map(str)
#     id_sou = all_edges_df['Source'].map(str)
#     id_tar = all_edges_df['Target'].map(str)
#     all_edges_df['Source_label'] = id_sou + '_' + var_sou
#     all_edges_df['Target_label'] = id_tar + '_' + var_tar
#     all_ver_list = list(all_edges_df['Source_label']) + list(
#         all_edges_df['Target_label'])
#     # set the unique of the vertexs
#     ver_list_unique = list(set(all_ver_list))
#     for v_id_var in ver_list_unique:
#         coupled_network.add_vertex(
#             v_id_var,
#             var_name=v_id_var.split('_')[1],
#             ga_id=v_id_var.split('_')[0],
#             label=v_id_var.split('_')[0],
#             size=30,
#             color=VAR_COLOR_DICT[v_id_var.split('_')[1]],
#             label_size=15)
#     # set all edges
#     tuples_es = [
#         tuple(x) for x in all_edges_df[['Source_label', 'Target_label']].values
#     ]
#     coupled_network.add_edges(tuples_es)
#     coupled_network.es['VarSou'] = list(all_edges_df['VarSou'])
#     coupled_network.es['VarTar'] = list(all_edges_df['VarTar'])
#     coupled_network.es['width'] = list(abs(all_edges_df['Strength'] * 1))
#     igraph.plot(coupled_network,
#                 BaseConfig.OUT_PATH +
#                 'Coupled_Network//Coupled_Network_ig.pdf',
#                 bbox=(1200, 1200),
#                 layout=coupled_network.layout('large'),
#                 margin=200)


def build_coupled_network():
    """
    build coupled network by the edges df from csv
    """
    all_edges_df = pd.read_csv(BaseConfig.OUT_PATH +
                               'Coupled_Network\\AllLinks.csv')
    filtered_edges_df = stretch_filter_links(all_edges_df)
    filtered_edges_df.to_csv(BaseConfig.OUT_PATH +
                             'Coupled_Network\\AllLinksFiltered.csv')
    coupled_network = build_net_by_links_df(filtered_edges_df)
    draw_net_on_map(coupled_network, 'coupled_network')

    # draw all inner and outer net
    var_index = 0
    for var in VARS_LIST:
        sec_var_index = 0
        for sec_var in VARS_LIST:
            if var == sec_var:
                sub_net = coupled_get_sub_inner_net(coupled_network, var)
                draw_net_on_map(sub_net, var + '_net')
            else:
                if sec_var_index > var_index:
                    sub_net = coupled_get_sub_outer_net(
                        coupled_network, var, sec_var)
                    draw_net_on_map(sub_net, var + '_and_' + sec_var + '_net')
            sec_var_index = sec_var_index + 1
        var_index = var_index + 1

    # coupled_draw_vars_network(coupled_network)
    # coupled_draw_agents_network(coupled_network)
    # from networkx.readwrite import json_graph
    # import json
    # data1 = json_graph.node_link_data(coupled_network)
    # filename = BaseConfig.OUT_PATH + 'Coupled_Network\\coupled_network.json'
    # with open(filename, 'w') as file_obj:
    #     json.dump(data1, file_obj)
    # nx.write_gexf(
    #     coupled_network,
    #     BaseConfig.OUT_PATH + 'Coupled_Network\\coupled_network.gexf')


def coupled_get_sub_inner_net(p_father_net, p_var_name):
    """
    get subgraph by the var name
    """
    # get nodes
    selected_ns = [
        n for n, d in p_father_net.nodes(data=True)
        if d['var_name'] == p_var_name
    ]
    sub_net = p_father_net.subgraph(selected_ns)
    return sub_net


def coupled_get_sub_outer_net(p_father_net, p_var_name_1, p_var_name_2):
    """
    get subgraph by the 2 var name
    """
    # get nodes
    selected_ns = [
        n for n, d in p_father_net.nodes(data=True)
        if d['var_name'] == p_var_name_1 or d['var_name'] == p_var_name_2
    ]
    sub_net = p_father_net.subgraph(selected_ns)
    return sub_net


def coupled_remove_inner_net(p_father_net):
    """
    remove the inner net
    """
    del_es = []
    for e in p_father_net.es:
        if e['VarSou'] == e['VarTar']:
            del_es.append(e)
    p_father_net.delete_edges(del_es)
    del_vs = []
    for v in p_father_net.vs:
        if p_father_net.degree(v) == 0:
            del_vs.append(v)
    p_father_net.delete_vertices(del_vs)
    return p_father_net


def coupled_draw_vars_network(p_coupled_network):
    """
    export vars network by quotient_graph
    """
    # group of nodes
    partitions = []
    for var in VARS_LIST:
        partitions.append([
            n for n, d in p_coupled_network.nodes(data=True)
            if d['var_name'] == var
        ])
    block_net = nx.quotient_graph(p_coupled_network, partitions, relabel=False)

    for n, d in block_net.nodes(data=True):
        var_label = list(n)[0].split('_')[1]
        block_net.nodes[n]['label'] = var_label

    fig = plt.figure(figsize=(10, 10), dpi=500)
    ax = plt.gca()
    pos = nx.circular_layout(block_net)
    for e in block_net.edges:
        ax.annotate("",
                    xy=pos[e[0]],
                    xycoords='data',
                    xytext=pos[e[1]],
                    textcoords='data',
                    arrowprops=dict(arrowstyle="->",
                                    color=VAR_COLOR_DICT[list(
                                        e[0])[0].split('_')[1]],
                                    shrinkA=3,
                                    shrinkB=3,
                                    patchA=None,
                                    patchB=None,
                                    connectionstyle="arc3,rad=rrr".replace(
                                        'rrr', str(0.005 * e[2]))))
    nx.draw_networkx_nodes(block_net,
                           pos,
                           node_size=[
                               block_net.nodes[n]['nedges'] * 30
                               for n, d in block_net.nodes(data=True)
                           ],
                           node_color=[
                               VAR_COLOR_DICT[list(n)[0].split('_')[1]]
                               for n, d in block_net.nodes(data=True)
                           ],
                           label=[
                               block_net.nodes[n]['label']
                               for n, d in block_net.nodes(data=True)
                           ])
    nx.draw_networkx_labels(block_net,
                            pos,
                            labels={
                                n: block_net.nodes[n]['label']
                                for n, d in block_net.nodes(data=True)
                            },
                            font_size=14,
                            font_color='#0007DA')
    plt.savefig(BaseConfig.OUT_PATH + 'Coupled_Network//vars_network.pdf')


def coupled_draw_agents_network(p_coupled_network):
    """
    export agents network by quotient_graph
    """
    all_edges_df = pd.read_csv(BaseConfig.OUT_PATH +
                               'Coupled_Network\\AllLinks.csv')
    id_sou = all_edges_df['Source'].map(str)
    id_tar = all_edges_df['Target'].map(str)
    all_ver_list = list(id_sou) + list(id_tar)
    # set the unique of the agents
    ver_list_unique = list(set(all_ver_list))
    # group of nodes
    partitions = []

    for a_id in ver_list_unique:
        partitions.append([
            n for n, d in p_coupled_network.nodes(data=True)
            if d['ga_id'] == a_id
        ])
    block_net = nx.quotient_graph(p_coupled_network, partitions, relabel=False)
    name_mapping = {}
    for b_node in block_net.nodes:
        name_mapping[b_node] = list(b_node)[0].split('_')[0]
        block_net.nodes[b_node]['ga_id'] = list(b_node)[0].split('_')[0]
        block_net.nodes[b_node]['color'] = 'r'
        block_net.nodes[b_node]['var_name'] = list(b_node)[0].split('_')[1]
    for e in block_net.edges:
        block_net.edges[e]['weight'] = 1
    nx.relabel_nodes(block_net, name_mapping)
    draw_net_on_map(block_net, 'Agents_Net')


def draw_net_on_map(p_network, p_net_name):
    """
    draw network on map on YR
    """
    fig = plt.figure(figsize=(20, 14), dpi=500)
    targetPro = ccrs.PlateCarree()
    ax = fig.add_subplot(1, 1, 1, projection=targetPro)
    # ax.set_global()
    # ax.stock_img()
    ax.add_feature(cfeature.OCEAN.with_scale('50m'))
    ax.add_feature(cfeature.LAND.with_scale('50m'))
    ax.add_feature(cfeature.RIVERS.with_scale('50m'))
    ax.add_feature(cfeature.LAKES.with_scale('50m'))

    ax.set_extent([95, 120, 31.5, 42])

    geoAgentShpPath = BaseConfig.GEO_AGENT_PATH + 'GA_WGS84.shp'

    geoAgentShp = ShapelyFeature(
        Reader(geoAgentShpPath).geometries(), ccrs.PlateCarree())
    ax.add_feature(geoAgentShp,
                   linewidth=0.5,
                   facecolor='None',
                   edgecolor='#4D5459',
                   alpha=0.8)

    centroid_data = pd.read_csv(BaseConfig.GEO_AGENT_PATH + 'GA_Centroid.csv',
                                index_col='GA_ID')
    pos = {}
    for n, d in p_network.nodes(data=True):
        # transform lon & lat
        lon = centroid_data.loc[int(d['ga_id'])]['longitude']
        lat = centroid_data.loc[int(d['ga_id'])]['latitude']
        mx, my = targetPro.transform_point(lon, lat, ccrs.PlateCarree())
        pos[n] = (mx, my)

    nx.draw_networkx_edges(
        p_network,
        pos=pos,
        width=[float(d['weight']) for (u, v, d) in p_network.edges(data=True)],
        alpha=0.5,
        # edge_color='k',
        edge_color=[
            VAR_COLOR_DICT[p_network.nodes[u]['var_name']]
            for (u, v, d) in p_network.edges(data=True)
        ],
        #    edge_color=[
        #        float(d['weight'])
        #        for (u, v, d) in p_network.edges(data=True)
        #    ],
        #    edge_cmap=plt.cm.Purples,
        arrows=True,
        arrowstyle='simple',
        arrowsize=2,
        connectionstyle="arc3,rad=0.05")
    nx.draw_networkx_nodes(
        p_network,
        pos=pos,
        node_size=[p_network.degree(n) * 5 for n in p_network],
        node_color=[d['color'] for n, d in p_network.nodes(data=True)])
    plt.savefig(BaseConfig.OUT_PATH + 'Coupled_Network//' + p_net_name +
                '.pdf')


def build_link_pcmci_noself(p_data_values, p_agent_names, p_var_sou,
                            p_var_tar):
    """
    build links by n column data
    """
    p_data_values = p_data_values.astype(float)
    [times_num, agent_num] = p_data_values.shape
    # set the data for PCMCI
    data_frame = pp.DataFrame(
        p_data_values,
        var_names=p_agent_names,
    )
    #   missing_flag=BaseConfig.BACKGROUND_VALUE)
    # new PCMCI
    pcmci = PCMCI(dataframe=data_frame,
                  cond_ind_test=ParCorr(significance='analytic'))
    # run PCMCI
    alpha_level = None
    results_pcmci = pcmci.run_pcmciplus(tau_min=0,
                                        tau_max=3,
                                        pc_alpha=alpha_level)
    # get the result
    graph_pcmci = results_pcmci['graph']
    q_matrix = results_pcmci['q_matrix']
    p_matrix = results_pcmci['p_matrix']
    val_matrix = results_pcmci['val_matrix']
    conf_matrix = results_pcmci['conf_matrix']
    ambiguous_triples = results_pcmci['ambiguous_triples']
    # filter these links
    links_df = pd.DataFrame(columns=('VarSou', 'VarTar', 'Source', 'Target',
                                     'TimeLag', 'Strength', 'Unoriented'))
    if graph_pcmci is not None:
        sig_links = (graph_pcmci != "") * (graph_pcmci != "<--")
    # q_matrix is the corrected p_matrix
    # elif q_matrix is not None:
    #     sig_links = (q_matrix <= alpha_level)
    # else:
    #     sig_links = (p_matrix <= alpha_level)
    for j in range(agent_num):
        links = {(p[0], -p[1]): np.abs(val_matrix[p[0], j, abs(p[1])])
                 for p in zip(*np.where(sig_links[:, j, :]))}
        # Sort by value
        sorted_links = sorted(links, key=links.get, reverse=True)
        for p in sorted_links:
            VarSou = p_var_sou
            VarTar = p_var_tar
            Source = p_agent_names[p[0]]
            Target = p_agent_names[j]
            TimeLag = p[1]
            Strength = val_matrix[p[0], j, abs(p[1])]
            Unoriented = None
            if graph_pcmci is not None:
                if graph_pcmci[j, p[0], 0] == "o-o":
                    Unoriented = 1
                    # "unoriented link"
                elif graph_pcmci[p[0], j, abs(p[1])] == "x-x":
                    Unoriented = 1
                    # "unclear orientation due to conflict"
                else:
                    Unoriented = 0
                links_df = links_df.append(pd.DataFrame({
                    'VarSou': [VarSou],
                    'VarTar': [VarTar],
                    'Source': [Source],
                    'Target': [Target],
                    'TimeLag': [TimeLag],
                    'Strength': [Strength],
                    'Unoriented': [Unoriented]
                }),
                                           ignore_index=True)
    # remove the self correlation edges
    links_df = links_df.loc[links_df['Source'] != links_df['Target']]
    return links_df


def stretch_filter_links(p_links):
    """
    filter links by some rules
    """
    # Stretch the data to fix monthly and yearly links
    only_monthly_vars = []
    only_yearly_vars = []
    for var in VARS_TIME_SCALE_DICT.keys():
        if VARS_TIME_SCALE_DICT[var] == 'yearly':
            only_yearly_vars.append(var)
        elif VARS_TIME_SCALE_DICT[var] == 'monthly_yearly':
            only_monthly_vars.append(var)
    links_only_monthly = p_links.loc[
        (p_links['VarSou'].isin(only_monthly_vars))
        & (p_links['VarTar'].isin(only_monthly_vars))]
    links_have_yearly = p_links.loc[(p_links['VarSou'].isin(only_yearly_vars))
                                    |
                                    (p_links['VarTar'].isin(only_yearly_vars))]

    # per98_of_mon_str = links_only_monthly['Strength'].describe(percentiles=[0.98]).loc['98%']
    # if per98_of_mon_str > 0.99:
    #     per98_of_mon_str = 0.97
    # per02_of_mon_str = links_only_monthly['Strength'].describe(percentiles=[0.02]).loc['2%']
    # links_only_monthly['Strength'][(links_only_monthly['Strength'] > per98_of_mon_str)] = per98_of_mon_str
    # links_only_monthly['Strength'][(links_only_monthly['Strength'] < per02_of_mon_str)] = per02_of_mon_str
    links_only_monthly['Strength'][(links_only_monthly['Strength'] > 0)] = (
        links_only_monthly['Strength'][(links_only_monthly['Strength'] > 0)] -
        links_only_monthly['Strength'][
            (links_only_monthly['Strength'] > 0)].min()) / (
                links_only_monthly['Strength'][
                    (links_only_monthly['Strength'] > 0)].max() -
                links_only_monthly['Strength'][
                    (links_only_monthly['Strength'] > 0)].min())
    links_only_monthly['Strength'][(
        links_only_monthly['Strength'] <
        0)] = (links_only_monthly['Strength'][
            (links_only_monthly['Strength'] < 0)].abs() -
               abs(links_only_monthly['Strength'][
                   (links_only_monthly['Strength'] < 0)].max())) / (
                       abs(links_only_monthly['Strength'][
                           (links_only_monthly['Strength'] < 0)].max()) -
                       abs(links_only_monthly['Strength'][
                           (links_only_monthly['Strength'] < 0)].min()))
    # (0.4-(0.1))/(0.1-(0.9))=-0.375
    # (0.6-(0.1))/(0.1-(0.9))=-0.625
    # per98_of_year_str = links_have_yearly['Strength'].describe(percentiles=[0.98]).loc['98%']
    # per02_of_year_str = links_have_yearly['Strength'].describe(percentiles=[0.02]).loc['2%']
    # if per02_of_year_str < -0.99:
    #     per02_of_year_str = -0.97
    # links_have_yearly['Strength'][(links_have_yearly['Strength'] > per98_of_year_str)] = per98_of_year_str
    # links_have_yearly['Strength'][(links_have_yearly['Strength'] < per02_of_year_str)] = per02_of_year_str
    links_have_yearly['Strength'][(links_have_yearly['Strength'] > 0)] = (
        links_have_yearly['Strength'][(links_have_yearly['Strength'] > 0)] -
        links_have_yearly['Strength'][
            (links_have_yearly['Strength'] > 0)].min()) / (
                links_have_yearly['Strength'][
                    (links_have_yearly['Strength'] > 0)].max() -
                links_have_yearly['Strength'][
                    (links_have_yearly['Strength'] > 0)].min())
    links_have_yearly['Strength'][(
        links_have_yearly['Strength'] <
        0)] = (links_have_yearly['Strength'][
            (links_have_yearly['Strength'] < 0)].abs() -
               abs(links_have_yearly['Strength'][
                   (links_have_yearly['Strength'] < 0)].max())) / (
                       abs(links_have_yearly['Strength'][
                           (links_have_yearly['Strength'] < 0)].max()) -
                       abs(links_have_yearly['Strength'][
                           (links_have_yearly['Strength'] < 0)].min()))
    stretched_links = pd.concat([links_only_monthly, links_have_yearly],
                                ignore_index=True)

    stretched_links.to_csv(BaseConfig.OUT_PATH + 'SelfNetworkCSV//' +
                           'SelfNetworkAll_stretched' + '.csv')

    strength_threshold = 0.2
    # select have oriented links
    filtered_links = stretched_links[(stretched_links['Unoriented'] == 0) & (
        (stretched_links['Strength'] > strength_threshold)
        | (stretched_links['Strength'] < -strength_threshold))].copy()
    return filtered_links


def build_net_by_links_df(p_edges_df):
    """
    built a net by a pands df
    """
    # build a net network
    network = nx.MultiDiGraph()
    # add every vertex to the net
    var_sou = p_edges_df['VarSou'].map(str)
    var_tar = p_edges_df['VarTar'].map(str)
    id_sou = p_edges_df['Source'].map(str)
    id_tar = p_edges_df['Target'].map(str)
    p_edges_df['Source_label'] = id_sou + '_' + var_sou
    p_edges_df['Target_label'] = id_tar + '_' + var_tar
    all_ver_list = list(p_edges_df['Source_label']) + list(
        p_edges_df['Target_label'])
    # set the unique of the vertexs
    ver_list_unique = list(set(all_ver_list))
    for v_id_var in ver_list_unique:
        network.add_node(v_id_var,
                         var_name=v_id_var.split('_')[1],
                         ga_id=v_id_var.split('_')[0],
                         label=v_id_var.split('_')[0],
                         size=30,
                         color=VAR_COLOR_DICT[v_id_var.split('_')[1]],
                         label_size=15)
    for lIndex, lRow in p_edges_df.iterrows():
        thisSou = lRow["Source_label"]
        thisTar = lRow["Target_label"]
        network.add_edge(thisSou,
                         thisTar,
                         weight=lRow['Strength'],
                         timelag=abs(lRow['TimeLag']))
        # for lf in p_edges_df.columns.values:
        #     inner_network.edges[thisSou, thisTar][lf] = lRow[lf]
    return network


def output_net_info_by_nx(p_networkx, p_path, abs=False):
    """
    use nx to output net info
    """
    # get the property of this network
    net_nodes = p_networkx.nodes()
    net_degree_zip = p_networkx.degree()
    net_degree = [v[1] for v in net_degree_zip]
    net_indeg_zip = p_networkx.in_degree()
    net_indeg = [v[1] for v in net_indeg_zip]
    net_oudeg_zip = p_networkx.out_degree()
    net_oudeg = [v[1] for v in net_oudeg_zip]
    net_out_div_in = np.divide(np.array(net_oudeg), np.array(net_indeg))

    i_net = igraph.Graph.from_networkx(p_networkx)
    net_degree_weighted = i_net.strength(mode='all', weights='strength')
    net_out_degree_weighted = i_net.strength(mode='out', weights='strength')
    net_in_degree_weighted = i_net.strength(mode='in', weights='strength')
    net_out_div_in_weighted = np.array(net_out_degree_weighted) / np.array(
        net_in_degree_weighted)
    net_betness = i_net.betweenness()
    net_betness_weighted = np.zeros(len(net_betness))
    if abs:
        net_betness_weighted = i_net.betweenness(weights='strength')
    net_cloness = i_net.closeness()
    net_pagerank = i_net.pagerank()

    # output information in a csv file
    out_net_info = pd.DataFrame({
        'Node': net_nodes,
        'degree': net_degree,
        'betweenness': net_betness,
        'betweenness_weighted': net_betness_weighted,
        'closeness': net_cloness,
        'pagerank': net_pagerank,
        'indegree': net_indeg,
        'outdegree': net_oudeg,
        'outdegree_div_indegree': net_out_div_in,
        'degree_weighted': net_degree_weighted,
        'out_degree_weighted': net_out_degree_weighted,
        'in_degree_weighted': net_in_degree_weighted,
        'out_div_in_weighted': net_out_div_in_weighted
    })
    out_net_info.to_csv(p_path)


def get_geo_distance(p_links, p_centroid_df):
    """
    get distance on earth
    @param p_links: links
    @param p_centroid_df: centroid info
    @return:
    """
    p_links['Distance'] = None
    for index, row in p_links.iterrows():
        thisSou = row["Source"]
        thisTar = row["Target"]
        souPoi = p_centroid_df[p_centroid_df["GA_ID"] == thisSou].copy()
        tarPoi = p_centroid_df[p_centroid_df["GA_ID"] == thisTar].copy()
        dist = geodesic(
            (souPoi.iloc[0]['latitude'], souPoi.iloc[0]['longitude']),
            (tarPoi.iloc[0]['latitude'], tarPoi.iloc[0]['longitude']))
        p_links.loc[index, 'Distance'] = dist.km
    return p_links


# 暂时弃用
def normalize_links_strength(p_self_net_df):
    yearly_vars = []
    monthly_vars = []
    for var in VARS_LIST:
        # monthly data
        if VARS_TIME_SCALE_DICT[var] == 'monthly_yearly':
            monthly_vars.append(var)
        # yearly data
        elif VARS_TIME_SCALE_DICT[var] == 'yearly':
            yearly_vars.append(var)
    yearly_links = p_self_net_df[
        (p_self_net_df['VarSou'].isin(yearly_vars))
        | (p_self_net_df['VarTar'].isin(yearly_vars))].copy()
    monthly_links = p_self_net_df[
        (p_self_net_df['VarSou'].isin(monthly_vars))
        & (p_self_net_df['VarTar'].isin(monthly_vars))].copy()

    max_min_scaler = lambda x: (x - np.min(x)) / (np.max(x) - np.min(x))

    yearly_links['Strength'] = yearly_links[['Strength']].apply(max_min_scaler)
    monthly_links['Strength'] = monthly_links[['Strength'
                                               ]].apply(max_min_scaler)
    both_links = yearly_links.append(monthly_links)
    print(both_links)
    return both_links


def main():
    print(time.strftime('%H:%M:%S', time.localtime(time.time())))
    # build_edges_to_csv()
    print(time.strftime('%H:%M:%S', time.localtime(time.time())))
    analyze_nets_to_file()
    print(time.strftime('%H:%M:%S', time.localtime(time.time())))
    # build_coupled_network()
    # print(time.strftime('%H:%M:%S', time.localtime(time.time())))
    print('Done! Thriller!')


# Run Code
# "self_" means funtions related to self_network
# "inner_" means funtion...........................................................s related to inner_network
# "outer_" means funtions related to outer_network
if __name__ == "__main__":

    main()

    # config = Config()
    # # 关系图中包括(include)哪些函数名。
    # # 如果是某一类的函数，例如类gobang，则可以直接写'gobang.*'，表示以gobang.开头的所有函数。（利用正则表达式）。
    # config.trace_filter = GlobbingFilter(include=[
    #     'main',
    #     'inner.*',
    #     'outer.*',
    #     'coupled.*',
    #     'build.*',
    #     'analyze.*',
    #     'draw.*',
    #     'output.*',
    # ])
    # # 该段作用是关系图中不包括(exclude)哪些函数。(正则表达式规则)
    # config.trace_filter = GlobbingFilter(exclude=[
    #     'pycallgraph.*',
    #     '*.secret_function',
    #     'FileFinder.*',
    #     'ModuleLockManager.*',
    #     'SourceFilLoader.*'
    # ])
    # graphviz = GraphvizOutput()
    # graphviz.output_file = 'CallGraph.png'
    # with PyCallGraph(output=graphviz, config=config):
    #     main()
