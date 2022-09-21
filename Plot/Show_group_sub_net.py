from heapq import merge
from attr import ib
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib
from matplotlib import cm
import matplotlib.pyplot as plt
from sklearn.manifold import smacof
from sympy import im

plt.rc('font', family='Arial')

SELF_GA_GROUP_TITLE = [
    'b  Mode 1', 'c  Mode 2', 'd  Mode 3', 'e  Mode 4', 'f  Mode 5'
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

SELF_GA_GROUP_LIST = [
    'GA_cluster_1', 'GA_cluster_2', 'GA_cluster_3', 'GA_cluster_4',
    'GA_cluster_5'
]

groups_show_shui_vars = ['WaterG', 'WaterS']
groups_show_shouhuo_vars = ['FruitG', 'IncomeR', 'CropG']
groups_show_touzi_vars = ['Fert', 'AgriM', 'RuralE']

OUT_PATH = 'D://YR_SDG_Network-Data//Output//'

fig = plt.figure(figsize=(15, 10))

stimulating_buffer = '#ffbd80'
stimulating_multiplier = '#ff7c67'
inhibiting_buffer = '#6feadf'
inhibiting_multiplier = '#5db9f1'

for nindex in np.arange(len(SELF_GA_GROUP_LIST)):
    linksbyType = pd.read_csv(OUT_PATH + 'SelfNetworkGrouply//' +
                              'Edges_of_linktype_' +
                              SELF_GA_GROUP_LIST[nindex] + '.csv')

    # build a net network
    network = nx.DiGraph()
    # add every vertex to the net
    all_ver_list = list(linksbyType['VarSou']) + list(linksbyType['VarTar'])
    # set the unique of the vertexs
    ver_list_unique = list(set(all_ver_list))
    b_m_fig_info = pd.read_csv(OUT_PATH + 'SelfNetworkGrouply' + '//' +
                               SELF_GA_GROUP_LIST[nindex] +
                               '_Self_buffer_multiplier_analysis_info.csv')

    node_color = '#fffaf3'
    for v_id_var in ver_list_unique:
        inter_info = b_m_fig_info[b_m_fig_info['Node'] == v_id_var]
        if inter_info['Interaction effect'].values > 1:
            if inter_info['Activity level of Stimulation'].values > 1:
                node_color = stimulating_multiplier
            else:
                node_color = stimulating_buffer
        else:
            if inter_info['Activity level of Inhibition'].values > 1:
                node_color = inhibiting_multiplier
            else:
                node_color = inhibiting_buffer

        network.add_node(v_id_var,
                         label=VAR_LABEL_DICT[v_id_var],
                         size=30,
                         color=node_color,
                         label_size=15)
    for lIndex, lRow in linksbyType.iterrows():
        thisSou = lRow["VarSou"]
        thisTar = lRow["VarTar"]
        network.add_edge(thisSou,
                         thisTar,
                         weight=lRow['Count'],
                         strength=lRow['Strength'])

    needshow_vars = groups_show_shui_vars + groups_show_shouhuo_vars + groups_show_touzi_vars

    sub_net_old = network.subgraph(needshow_vars)
    sub_net = sub_net_old.copy()
    # for a_bunch in needshow_vars[4:8]:
    #     for b_bunch in needshow_vars[4:8]:
    #         if a_bunch != b_bunch:
    #             sub_net.remove_edges_from([(a_bunch, b_bunch)])

    #### draw graph ####
    # fig, ax = plt.subplots(figsize=(10, 10))
    ax = fig.add_subplot(2, 3, nindex + 2)
    # pos = nx.circular_layout(sub_net)

    # pos = {
    #     'IncomeR': (0, -0.26),
    #     'CropG': (-0.3, 0.26),
    #     'FruitG': (0.3, 0.26),
    # }
    pos_shui = [(-0.2, -0.5), (0.2, -0.5)]
    pos = dict(zip(groups_show_shui_vars, pos_shui))

    pos_shouhuo = [(-0.4, 0), (0, 0), (0.4, 0)]
    pos_add_shouhuo = dict(zip(groups_show_shouhuo_vars, pos_shouhuo))
    pos.update(pos_add_shouhuo)

    pos_touzi = [(-0.4, 0.5), (0, 0.5), (0.4, 0.5)]
    pos_add_touzi = dict(zip(groups_show_touzi_vars, pos_touzi))
    pos.update(pos_add_touzi)

    # edge_color = [
    #     float(d['strength']) for (u, v, d) in sub_net.edges(data=True)
    # ]
    # cmap = plt.get_cmap('coolwarm')
    # norm = matplotlib.colors.Normalize(vmin=-1,
    #                                    vmax=1)
    # sm = plt.cm.ScalarMappable(cmap=cmap,norm=norm)

    edge_color = []
    for u, v, d in sub_net.edges(data=True):
        strength = float(d['strength'])
        if strength < 0:
            edge_color.append(cm.Blues(-strength))
        else:
            edge_color.append(cm.Reds(strength))

    edge_width_v = [
        float(d['weight']) for (u, v, d) in sub_net.edges(data=True)
    ]
    edge_width = 2 + 3 * (np.array(edge_width_v) - np.array(edge_width_v).min(
    )) / (np.array(edge_width_v).max() - np.array(edge_width_v).min())
    # edge_width = []
    # for u, v, d in sub_net.edges(data=True):
    #     width = float(d['weight']) * 1.6
    #     if width > 5:
    #         edge_width.append(5)
    #     else:
    #         edge_width.append(width)
    # nx.draw_networkx(
    #     sub_net,
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
        sub_net,
        pos,
        node_size=1500,
        node_color=[d['color'] for n, d in sub_net.nodes(data=True)],
        # cmap=plt.cm.Wistia,
        edgecolors='#d9d9d9',
        alpha=1)

    nx.draw_networkx_edges(
        sub_net,
        pos,
        width=edge_width,
        alpha=0.7,
        edge_color=edge_color,
        #    edge_cmap=plt.cm.coolwarm,
        arrows=True,
        arrowstyle='->',
        min_target_margin=20,
        arrowsize=8,
        connectionstyle="arc3,rad=0.1")

    nx.draw_networkx_labels(
        sub_net,
        pos=pos,
        labels={
            n: sub_net.nodes[n]['label']
            for n, d in sub_net.nodes(data=True)
        },
        font_size=12,
        font_color='k',
        # font_family='SimHei',
        font_family='Arial',
        font_weight='bold')
    ax.set_title(SELF_GA_GROUP_TITLE[nindex],
                 loc='left',
                 size=16,
                 fontweight='bold')
    plt.axis("off")

axleg = fig.add_subplot(2, 3, 1)
axleg.scatter([0.6, 0.6, 0.6, 0.6], [0.4, 0.3, 0.2, 0.1],
              marker='o',
              color=[
                  stimulating_buffer, stimulating_multiplier,
                  inhibiting_buffer, inhibiting_multiplier
              ],
              s=[350, 350, 350, 350],
              alpha=1,
              edgecolors='#d9d9d9',
              zorder=100)
axleg.set_xlim(0, 1)
axleg.set_ylim(0, 1)
# axleg.text(0.06,
#            0.1,
#            'Stimulation buffer',
#            fontsize=12,
#            ha='left',
#            va='center')
# axleg.text(0.06,
#            0.03,
#            'Stimulation multiplier',
#            fontsize=12,
#            ha='left',
#            va='center')
# axleg.text(0.5,
#            0.1,
#            'Inhibition buffer',
#            fontsize=12,
#            ha='left',
#            va='center')
# axleg.text(0.5,
#            0.03,
#            'Inhibition multiplier',
#            fontsize=12,
#            ha='left',
#            va='center')
axleg.set_title('a Agricultural development modes',
                loc='left',
                size=16,
                fontweight='bold')
axleg.axis("off")

fig.tight_layout()
plt.savefig(OUT_PATH + 'Figures//Sub_Linktype_5_groups.pdf')