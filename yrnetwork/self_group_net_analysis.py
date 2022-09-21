from ast import Return
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

import coupled_network_causal_of_SDG as mainFuc

plt.rc('font', family='Arial')

SELF_GA_GROUP_DICT = {
    'GA_cluster_1': '#006100',
    'GA_cluster_2': '#7aab00',
    'GA_cluster_3': '#ffff00',
    'GA_cluster_4': '#ff9900',
    'GA_cluster_5': '#ff2200',
}
SELF_GA_GROUP_TITLE = [
    'a  Mode 1', 'b  Mode 2', 'c  Mode 3', 'd  Mode 4', 'e  Mode 5'
]
SELF_GA_GROUP_LIST = [
    'GA_cluster_1', 'GA_cluster_2', 'GA_cluster_3', 'GA_cluster_4',
    'GA_cluster_5'
]

VAR_LABEL_DICT = {
    'CO2': 'CO2',
    'Temp': 'Temp',
    'VPD': 'VPD',
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


def self_draw_group_nets(p_edges_df):
    fig_b_m_1 = plt.figure(figsize=(22, 22))
    fig_b_m_2 = plt.figure(figsize=(22, 22))
    for nindex in np.arange(len(SELF_GA_GROUP_LIST)):
        ga_group_data = pd.read_csv(BaseConfig.GEO_AGENT_PATH +
                                    SELF_GA_GROUP_LIST[nindex] + '.csv')
        ga_group_df = p_edges_df[(p_edges_df['Source'].isin(
            list(ga_group_data['GA_ID'])))]
        mainFuc.self_draw_links_df_in_linktype(
            ga_group_df, SELF_GA_GROUP_LIST[nindex],
            SELF_GA_GROUP_DICT[SELF_GA_GROUP_LIST[nindex]])
        mainFuc.self_draw_net_in_one_and_pos_neg_parts(
            ga_group_df, 'SelfNetworkGrouply', SELF_GA_GROUP_LIST[nindex])
        out_self_buffer_multiplier_analysis_info_group('SelfNetworkGrouply',
                                                       nindex)

        fig_b_m_1 = self_buffer_multiplier_analysis_for_group_1(
            'SelfNetworkGrouply', nindex, fig_b_m_1)
        ax_leg1 = fig_b_m_1.add_subplot(3, 2, 6)
        ax_leg1.scatter([2.5, 2.5], [3.05, 2.6],
                        marker='o',
                        color=['#ffffff', '#ffffff'],
                        s=[2 * 500, 1 * 500],
                        alpha=0.8,
                        edgecolors='#595c5d',
                        linewidths=1,
                        zorder=100)
        ax_leg1.text(2.7,
                     2.54,
                     'betweenness = 2 \n\nbetweenness = 1 ',
                     fontsize=16)
        ax_leg1.set_xlim(0, 6)
        ax_leg1.set_ylim(0, 5)
        ax_leg1.axis("off")

        fig_b_m_1.savefig(BaseConfig.OUT_PATH + 'SelfNetworkGrouply' + '//5G' +
                          '_Self_buffer_multiplier_analysis_1.pdf',
                          dpi=500,
                          bbox_inches='tight')
        fig_b_m_2 = self_buffer_multiplier_analysis_for_group_2(
            'SelfNetworkGrouply', nindex, fig_b_m_2)
        ax_leg2 = fig_b_m_2.add_subplot(3, 2, 6)
        ax_leg2.scatter([1.45, 1.45, 3.95, 3.95], [2.05, 2.5, 2.05, 2.5],
                        marker='o',
                        color=['#69acd6', '#fa6648', '#ffffff', '#ffffff'],
                        s=[3 * 250, 3 * 250, 2 * 250, 4 * 250],
                        alpha=0.8,
                        edgecolors='#595c5d',
                        linewidths=1,
                        zorder=100)
        ax_leg2.text(
            1.6,
            2,
            ' Stimulation sub-network              betweenness = 4 \n\n Inhibition sub-network                 betweenness = 2 ',
            fontsize=16)
        ax_leg2.set_xlim(0, 6)
        ax_leg2.set_ylim(0, 5)
        ax_leg2.axis("off")
        fig_b_m_2.savefig(BaseConfig.OUT_PATH + 'SelfNetworkGrouply' + '//5G' +
                          '_Self_buffer_multiplier_analysis_2.pdf',
                          dpi=500,
                          bbox_inches='tight')


def out_self_buffer_multiplier_analysis_info_group(p_data_folder,
                                                   p_group_index):
    network_info = pd.read_csv(BaseConfig.OUT_PATH + p_data_folder + '//' +
                               SELF_GA_GROUP_LIST[p_group_index] +
                               '_Self_oneNet_info.csv')
    network_pos_info = pd.read_csv(BaseConfig.OUT_PATH + p_data_folder + '//' +
                                   SELF_GA_GROUP_LIST[p_group_index] +
                                   '_Self_network_pos_info.csv')
    network_neg_info = pd.read_csv(BaseConfig.OUT_PATH + p_data_folder + '//' +
                                   SELF_GA_GROUP_LIST[p_group_index] +
                                   '_Self_network_neg_info.csv')
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
        BaseConfig.OUT_PATH + p_data_folder + '//' +
        SELF_GA_GROUP_LIST[p_group_index] +
        '_Self_buffer_multiplier_analysis_info.csv')


def self_buffer_multiplier_analysis_for_group_1(p_data_folder, p_group_index,
                                                p_fig_b_m_1):
    ax1 = p_fig_b_m_1.add_subplot(3, 2, p_group_index + 1)
    # read csv
    network_info = pd.read_csv(BaseConfig.OUT_PATH + p_data_folder + '//' +
                               SELF_GA_GROUP_LIST[p_group_index] +
                               '_Self_oneNet_info.csv')
    network_pos_info = pd.read_csv(BaseConfig.OUT_PATH + p_data_folder + '//' +
                                   SELF_GA_GROUP_LIST[p_group_index] +
                                   '_Self_network_pos_info.csv')
    network_neg_info = pd.read_csv(BaseConfig.OUT_PATH + p_data_folder + '//' +
                                   SELF_GA_GROUP_LIST[p_group_index] +
                                   '_Self_network_neg_info.csv')

    # # fig0, (ax1, ax2) = plt.subplots(1, 2, sharey=True,figsize=(14, 8), dpi=500)
    # fig = plt.figure(figsize=(11, 6), dpi=500)
    # # fig = plt.figure(figsize=(14, 8), dpi=500)
    # rect1 = [0, 0, 1, 1]  # [左, 下, 宽, 高] 规定的矩形区域 （全部是0~1之间的数，表示比例）

    # ax1 = plt.axes(rect1)

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
                          3,
                          1,
                          facecolor="#f47d6a",
                          alpha=0.4,
                          clip_on=False))
    #left up
    ax1.add_patch(
        patches.Rectangle((0, 1),
                          1,
                          4,
                          facecolor="#7eb9d7",
                          alpha=0.8,
                          clip_on=False))
    #right up
    ax1.add_patch(
        patches.Rectangle((1, 1),
                          3,
                          4,
                          facecolor="#f47d6a",
                          alpha=0.8,
                          clip_on=False))

    ax1.axvline(x=1, linewidth=2, color='#ffffff', linestyle='--', zorder=1)
    ax1.axhline(y=1, linewidth=2, color='#ffffff', linestyle='--', zorder=1)

    sizes = []
    for n in network_info['betweenness']:
        if n < 1:
            sizes.append(1 * 500)
        else:
            sizes.append(n * 500)

    ax1.scatter(network_pos_info['degree_weighted'] /
                network_neg_info['degree_weighted'],
                network_info['outdegree_div_indegree'],
                marker='o',
                color='#ffffff',
                s=sizes,
                alpha=0.8,
                edgecolors='#595c5d',
                linewidths=1,
                label='  betweenness = 4 \n\n  betweenness = 2 ',
                zorder=2)
    for x, y, text in zip(
            network_pos_info['degree_weighted'] /
            network_neg_info['degree_weighted'],
            network_info['outdegree_div_indegree'], network_info['Node']):
        if x > 6 or y > 6:
            continue
        ax1.text(x,
                 y,
                 VAR_LABEL_DICT[text],
                 fontsize=14,
                 ha='center',
                 va='center',
                 color='#000000')

    ax1.set_xlim(0, 4)
    ax1.set_ylim(0, 5)

    # ax1.spines['right'].set_visible(False)

    # ax1.set_xticks([0.2, 0.5, 1, 1.5, 2])
    # label_X1 = ['0.2', '0.5', '1', '1.5', '2']
    # ax1.set_xticklabels(label_X1, fontsize=14, fontfamily='Arial')

    plt.yticks(fontproperties='Arial', size=14)
    plt.xticks(fontproperties='Arial', size=14)
    ax1.set_xlabel(
        '                    Interaction effect\n                    (Stimulation weighted degree / Inhibition weighted degree)',
        fontsize=16)
    ax1.set_ylabel('Driving effect (overall outdegree / overall indegree)',
                   fontsize=16)
    # ax1.legend(loc='upper left',
    #            ncol=1,
    #            prop={
    #                'size': 14,
    #            },
    #            labelspacing=1,
    #            borderpad=0.8,
    #            handletextpad=0.1,
    #            markerscale=0.0000000000001,
    #            frameon=True)
    # ax1.scatter([0.3, 0.3], [5.605, 5.205],
    #             marker='o',
    #             color=['#ffffff', '#ffffff'],
    #             s=[4 * 200, 2 * 200],
    #             alpha=0.8,
    #             edgecolors='#595c5d',
    #             linewidths=1,
    #             zorder=100)

    # ax1.text(0.4,
    #          -0.61,
    #          '← Inhibiting',
    #          fontsize=16,
    #          c='#000000',
    #          style='italic',
    #          ha='center',
    #          va='center')
    # ax1.text(2.27,
    #          -0.61,
    #          'Stimulating →',
    #          fontsize=16,
    #          c='#000000',
    #          style='italic',
    #          ha='center',
    #          va='center')
    ax1.set_title(SELF_GA_GROUP_TITLE[p_group_index], loc='left', size=16)
    return p_fig_b_m_1


def self_buffer_multiplier_analysis_for_group_2(p_data_folder, p_group_index,
                                                p_fig_b_m_2):
    ax2 = p_fig_b_m_2.add_subplot(3, 2, p_group_index + 1)
    # read csv
    network_info = pd.read_csv(BaseConfig.OUT_PATH + p_data_folder + '//' +
                               SELF_GA_GROUP_LIST[p_group_index] +
                               '_Self_oneNet_info.csv')
    network_pos_info = pd.read_csv(BaseConfig.OUT_PATH + p_data_folder + '//' +
                                   SELF_GA_GROUP_LIST[p_group_index] +
                                   '_Self_network_pos_info.csv')
    network_neg_info = pd.read_csv(BaseConfig.OUT_PATH + p_data_folder + '//' +
                                   SELF_GA_GROUP_LIST[p_group_index] +
                                   '_Self_network_neg_info.csv')
    # fig = plt.figure(figsize=(11, 6), dpi=500)
    # # ax = fig.add_subplot(1, 1, 1)
    # rect = [0, 0, 1, 1]  # [左, 下, 宽, 高] 规定的矩形区域 （全部是0~1之间的数，表示比例）
    # ax = plt.axes(rect)

    ax2.grid(color='#ffffff', zorder=-10)
    ax2.axvspan(0.01, 1, facecolor='#7e968e', alpha=1, zorder=-10)
    ax2.axvspan(1, 100, facecolor='#cfe1e8', alpha=1, zorder=-10)
    ax2.axvline(x=1, linewidth=2, color='#ffffff', linestyle='--', zorder=1)
    # ax2.axvline(x=1, linewidth=2, color='#b2735e', linestyle='--', zorder=-10)

    # x_c = 0.1
    # y_c = 112.5
    # plt.text(x_c,
    #          y_c + 18,
    #          'buffers',
    #          fontsize=26,
    #          c='#242b31',
    #          style='italic',
    #          ha='center',
    #          va='center')
    # G_buffer = nx.DiGraph()
    # G_buffer.add_nodes_from([0, 1, 2, 3, 4, 5])
    # G_buffer.add_edges_from([(0, 4), (1, 4), (2, 4), (3, 4), (4, 5)])

    # pos_buffer = {
    #     0: np.array([x_c - 0.047, y_c + 13]),
    #     1: np.array([x_c - 0.05, y_c + 5]),
    #     2: np.array([x_c - 0.05, y_c - 5]),
    #     3: np.array([x_c - 0.047, y_c - 13]),
    #     4: np.array([x_c, y_c]),
    #     5: np.array([x_c + 0.095, y_c])
    # }
    # nx.draw_networkx_edges(G_buffer,
    #                        pos_buffer,
    #                        ax=ax2,
    #                        width=1,
    #                        alpha=1,
    #                        edge_color='k',
    #                        arrows=True,
    #                        arrowstyle='->',
    #                        arrowsize=10,
    #                        node_size=600)
    # nx.draw_networkx_nodes(G_buffer,
    #                        pos_buffer,
    #                        ax=ax2,
    #                        nodelist=[4],
    #                        node_size=600,
    #                        node_color='#ffffff',
    #                        edgecolors='#000000',
    #                        alpha=1)
    # ax2.set_axis_on()
    # ax2.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)

    # x_c = 10
    # y_c = 112.5
    # plt.text(x_c,
    #          y_c + 18,
    #          'multipliers',
    #          fontsize=26,
    #          c='#242b31',
    #          style='italic',
    #          ha='center',
    #          va='center')
    # G_multi = nx.DiGraph()
    # G_multi.add_nodes_from([0, 1, 2, 3, 4, 5])
    # G_multi.add_edges_from([(0, 1), (1, 2), (1, 3), (1, 4), (1, 5)])

    # pos_multi = {
    #     0: np.array([x_c - 5, y_c]),
    #     1: np.array([x_c, y_c]),
    #     2: np.array([x_c + 8, y_c + 13]),
    #     3: np.array([x_c + 10, y_c + 5]),
    #     4: np.array([x_c + 10, y_c - 5]),
    #     5: np.array([x_c + 8, y_c - 13])
    # }
    # nx.draw_networkx_edges(G_multi,
    #                        pos_multi,
    #                        ax=ax2,
    #                        width=1,
    #                        alpha=1,
    #                        edge_color='k',
    #                        arrows=True,
    #                        arrowstyle='->',
    #                        arrowsize=10,
    #                        node_size=600)
    # nx.draw_networkx_nodes(G_multi,
    #                        pos_multi,
    #                        ax=ax2,
    #                        nodelist=[1],
    #                        node_size=600,
    #                        node_color='#ffffff',
    #                        edgecolors='#000000',
    #                        alpha=1)
    # ax2.set_axis_on()
    # ax2.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)

    # ax2.set_xscale('symlog', linthresh=0.05)
    ax2.set_xscale('log')

    network_neg_info = network_neg_info.sort_values(by='betweenness',
                                                    ascending=False)
    sizes = []
    for n in network_neg_info['betweenness']:
        if n < 1:
            sizes.append(1 * 250)
        else:
            sizes.append(n * 250)
    ax2.scatter(network_neg_info['out_div_in_weighted'],
                network_neg_info['degree_weighted'],
                marker='o',
                color='#69acd6',
                s=sizes,
                alpha=0.8,
                edgecolors='#595c5d',
                linewidths=1,
                label='Inhibition sub-network             betweenness = 2 ',
                zorder=2)
    for x, y, text in zip(network_neg_info['out_div_in_weighted'],
                          network_neg_info['degree_weighted'],
                          network_neg_info['Node']):
        if x < 0.01 or x > 100:
            continue
        ax2.text(x,
                 y,
                 VAR_LABEL_DICT[text],
                 fontsize=14,
                 ha='center',
                 va='center',
                 color='#042145')

    network_pos_info = network_pos_info.sort_values(by='betweenness',
                                                    ascending=False)
    sizes = []
    for n in network_pos_info['betweenness']:
        if n < 1:
            sizes.append(1 * 250)
        else:
            sizes.append(n * 250)
    ax2.scatter(network_pos_info['out_div_in_weighted'],
                network_pos_info['degree_weighted'],
                marker='o',
                color='#fa6648',
                s=sizes,
                alpha=0.8,
                edgecolors='#595c5d',
                linewidths=1,
                label='Stimulation sub-network          betweenness = 4 ',
                zorder=2)
    for x, y, text in zip(network_pos_info['out_div_in_weighted'],
                          network_pos_info['degree_weighted'],
                          network_pos_info['Node']):
        if x < 0.01 or x > 100:
            continue
        ax2.text(x,
                 y,
                 VAR_LABEL_DICT[text],
                 fontsize=14,
                 ha='center',
                 va='center',
                 color='#450208')

    # ax2.legend(loc='upper left',
    #           ncol=1,
    #           prop={
    #               'size': 14,
    #           },
    #           labelspacing=1,
    #           borderpad=0.8,
    #           handletextpad=0.1,
    #           markerscale=0.0000000001,
    #           frameon=True)
    # plt.scatter([0.0405, 0.0405, 0.21, 0.21], [155, 165, 155, 165],
    #             marker='o',
    #             color=['#fa6648', '#69acd6', '#ffffff', '#ffffff'],
    #             s=[3 * 120, 3 * 120, 2 * 120, 4 * 120],
    #             alpha=0.8,
    #             edgecolors='#595c5d',
    #             linewidths=1,
    #             zorder=100)

    # leftlim = math.pow(10, -1.5)
    # rightlim = math.pow(10, 1.5)
    ax2.set_xlim(0.01, 100)
    # ax.set_ylim(0, 175)
    ax2.set_xticks([0.01, 0.1, 1, 10, 100])
    # ax.set_yticks([0, 25, 50, 75, 100, 125, 150, 175])
    label_X = ['0.01', '0.1', '1', '10', '100']
    ax2.set_xticklabels(label_X, fontsize=14, fontfamily='Arial')
    plt.yticks(fontproperties='Arial', size=14)
    plt.xticks(fontproperties='Arial', size=14)
    ax2.set_xlabel('Activity level\n(weighted outdegree / weighted indegree)',
                   fontsize=16)
    ax2.set_ylabel('Interconnectedness (weighted degree)', fontsize=16)
    # plt.text(0.1,
    #          -12,
    #          '← passive',
    #          fontsize=16,
    #          c='#000000',
    #          style='italic',
    #          ha='center',
    #          va='center')
    # plt.text(10,
    #          -12,
    #          'active →',
    #          fontsize=16,
    #          c='#000000',
    #          style='italic',
    #          ha='center',
    #          va='center')
    ax2.set_title(SELF_GA_GROUP_TITLE[p_group_index], loc='left', size=16)
    return p_fig_b_m_2