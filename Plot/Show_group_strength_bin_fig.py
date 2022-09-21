from cProfile import label
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib
from matplotlib import cm
import matplotlib.pyplot as plt
from pytest import mark

plt.rc('font', family='Arial')
colors = ['#006014', '#7aa92c', '#fffc47', '#ff9729', '#ff200e']
SELF_GA_GROUP_LIST = [
    'GA_cluster_1', 'GA_cluster_2', 'GA_cluster_3', 'GA_cluster_4',
    'GA_cluster_5'
]
SELF_GA_GROUP_NAME_LIST = ['Mode 1', 'Mode 2', 'Mode 3', 'Mode 4', 'Mode 5']
OUT_PATH = 'D://YR_SDG_Network-Data//Output//'
GEO_AGENT_PATH = 'D://YR_SDG_Network//data//GeoAgent//'

self_edges_df = pd.read_csv(OUT_PATH +
                            'SelfNetworkCSV//SelfNetworkAll_stretched.csv')
p_edges_df = self_edges_df[(self_edges_df['Unoriented'] == 0)].copy()
p_edges_df_pos = p_edges_df[p_edges_df['Strength'] >= 0]
p_edges_df_neg = p_edges_df[p_edges_df['Strength'] < 0]

fig = plt.figure(figsize=(15, 10), dpi=500)
rect1 = [0, 0.5, 1, 0.5]  # [左, 下, 宽, 高] 规定的矩形区域 （全部是0~1之间的数，表示比例）
rect2 = [0, 0, 1, 0.5]
ax_pos = plt.axes(rect1)
ax_neg = plt.axes(rect2)

ax_pos.grid(axis='y')
ax_neg.grid(zorder=-100, axis='y')

# ax.boxplot(p_edges_df['Strength'],
#               positions=[1])

# for nindex in np.arange(len(SELF_GA_GROUP_LIST)):
#     ga_group_data = pd.read_csv(GEO_AGENT_PATH + SELF_GA_GROUP_LIST[nindex] +
#                                 '.csv')
#     ga_group_df = p_edges_df[(p_edges_df['Source'].isin(
#         list(ga_group_data['GA_ID'])))]
#     ax.boxplot(ga_group_df['Strength'],
#                   positions=[nindex + 2])
# plt.savefig(OUT_PATH + 'Figures//Sub_4_groups_strength_bins.pdf')

medianprops = dict(color="black")
meanprops = dict(color='red', linestyle='-', linewidth=1.5)

parts = ax_pos.violinplot(p_edges_df_pos['Strength'],
                          positions=[1],
                          showmeans=False,
                          showmedians=False,
                          showextrema=False)
for pc in parts['bodies']:
    pc.set_facecolor('#fa6648')
    # pc.set_edgecolor('black')
    pc.set_alpha(0.3)
    # parts['cmeans'].set_color('black')
ax_pos.scatter(np.ones(p_edges_df_pos['Strength'].count()) * 1,
               p_edges_df_pos['Strength'],
               color='#8b898b',
               alpha=0.4)
ax_pos.boxplot(p_edges_df_pos['Strength'],
               positions=[1],
               medianprops=medianprops,
               meanprops=meanprops,
               meanline=True,
               showmeans=True,
               labels=['labels'])
ax_pos.text(
    1,
    1.035,
    'Count:' + str(p_edges_df_pos['Strength'].count()),
    horizontalalignment='center',
    color='black',
    # weight='bold',
    family='Arial',
    size=14)

parts = ax_neg.violinplot(p_edges_df_neg['Strength'],
                          positions=[1],
                          showmeans=False,
                          showmedians=False,
                          showextrema=False)
for pc in parts['bodies']:
    pc.set_facecolor('#69acd6')
    # pc.set_edgecolor('black')
    pc.set_alpha(0.4)
    # parts['cmeans'].set_color('black')
ax_neg.scatter(np.ones(p_edges_df_neg['Strength'].count()) * 1,
               p_edges_df_neg['Strength'],
               color='#8b898b',
               alpha=0.4)
ax_neg.boxplot(p_edges_df_neg['Strength'],
               positions=[1],
               medianprops=medianprops,
               meanprops=meanprops,
               meanline=True,
               showmeans=True)
ax_neg.text(
    1,
    -1.065,
    'Count:' + str(p_edges_df_neg['Strength'].count()),
    horizontalalignment='center',
    color='black',
    # weight='bold',
    family='Arial',
    size=14)

for nindex in np.arange(len(SELF_GA_GROUP_LIST)):
    ga_group_data = pd.read_csv(GEO_AGENT_PATH + SELF_GA_GROUP_LIST[nindex] +
                                '.csv')
    ga_group_df_pos = p_edges_df_pos[(p_edges_df_pos['Source'].isin(
        list(ga_group_data['GA_ID'])))]
    parts = ax_pos.violinplot(ga_group_df_pos['Strength'],
                              positions=[nindex + 2],
                              showmeans=False,
                              showmedians=False,
                              showextrema=False)
    for pc in parts['bodies']:
        pc.set_facecolor('#fa6648')
        # pc.set_edgecolor('black')
        pc.set_alpha(0.3)
        # parts['cmeans'].set_color('black')
    ax_pos.scatter(np.ones(ga_group_df_pos['Strength'].count()) * (nindex + 2),
                   ga_group_df_pos['Strength'],
                   color=colors[nindex],
                   alpha=0.4)
    print(SELF_GA_GROUP_LIST[nindex] + 'mean Strength pos')
    print(ga_group_df_pos['Strength'].mean())
    ax_pos.boxplot(ga_group_df_pos['Strength'],
                   positions=[nindex + 2],
                   medianprops=medianprops,
                   meanprops=meanprops,
                   meanline=True,
                   showmeans=True)
    ax_pos.text(
        nindex + 2,
        1.035,
        'Count:' + str(ga_group_df_pos['Strength'].count()),
        horizontalalignment='center',
        color='black',
        # weight='bold',
        family='Arial',
        size=14)

    ga_group_df_neg = p_edges_df_neg[(p_edges_df_neg['Source'].isin(
        list(ga_group_data['GA_ID'])))]
    parts = ax_neg.violinplot(ga_group_df_neg['Strength'],
                              positions=[nindex + 2],
                              showmeans=False,
                              showmedians=False,
                              showextrema=False)
    for pc in parts['bodies']:
        pc.set_facecolor('#69acd6')
        # pc.set_edgecolor('black')
        pc.set_alpha(0.4)
        # parts['cmeans'].set_color('black')
    ax_neg.scatter(np.ones(ga_group_df_neg['Strength'].count()) * (nindex + 2),
                   ga_group_df_neg['Strength'],
                   color=colors[nindex],
                   alpha=0.4)
    print(SELF_GA_GROUP_LIST[nindex] + 'mean Strength neg')
    print(ga_group_df_neg['Strength'].mean())
    ax_neg.boxplot(ga_group_df_neg['Strength'],
                   positions=[nindex + 2],
                   medianprops=medianprops,
                   meanprops=meanprops,
                   meanline=True,
                   showmeans=True)
    ax_neg.text(
        nindex + 2,
        -1.065,
        'Count:' + str(ga_group_df_neg['Strength'].count()),
        horizontalalignment='center',
        color='black',
        # weight='bold',
        family='Arial',
        size=14)

ax_pos.set_ylim(0, 1.1)
ax_neg.set_ylim(-1.1, 0)
# ax_neg.set_xticks([1, 2, 3, 4, 5])
# label_x_neg = ['Overall', 'Pattern 1', 'Pattern 2', 'Pattern 3', 'Pattern 4']
# ax_neg.set_xticklabels(label_x_neg, fontsize=14, fontfamily='Arial')

ax_pos.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1])
label_y_pos = ['0', '0.2', '0.4', '0.6', '0.8', '1']
ax_pos.set_yticklabels(label_y_pos, fontsize=14, fontfamily='Arial')

ax_neg.set_yticks([-1, -0.2, -0.4, -0.6, -0.8, 0])
label_y_neg = ['-1', '-0.2', '-0.4', '-0.6', '-0.8', '0']
ax_neg.set_yticklabels(label_y_neg, fontsize=14, fontfamily='Arial')

label_x_neg = ['Basin', 'Mode 1', 'Mode 2', 'Mode 3', 'Mode 4', 'Mode 5']
ax_neg.set_xticklabels(label_x_neg, fontsize=14, fontfamily='Arial')

ax_neg.set_xlabel('Different areas', fontsize=16)
ax_pos.set_ylabel('Strength', fontsize=16)
ax_neg.set_ylabel('Strength', fontsize=16)

# ax_pos.legend(loc=0)

plt.savefig(OUT_PATH + 'Figures//Sub_5_groups_strength_violinplot.pdf',
            dpi=500,
            bbox_inches='tight')
