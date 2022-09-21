from cProfile import label
import numpy as np
import pandas as pd
import matplotlib
from matplotlib import cm
import matplotlib.pyplot as plt
import scipy.stats as st
from matplotlib.colors import ListedColormap, BoundaryNorm
from pandas.api.types import CategoricalDtype

plt.rc('font', family='Arial')

def main():
    OUT_PATH = 'D://YR_SDG_Network-Data//Output//'
    df_edge_direct = pd.read_csv(OUT_PATH + 'SelfNetworkGrouply//' +
                                 'Edges_of_linktype_All_edges.csv')
    df_edge_indirect = pd.read_csv(
        OUT_PATH + 'SelfNetworkGrouply//' +
        'Edges_of_linktype_All_edges_indirect_strength.csv')
    x_vars = ['Fert', 'AgriM', 'RuralE']
    y_vars = [
        'CropG', 'FruitG', 'IncomeR', 'Fert', 'AgriM', 'RuralE', 'WaterG',
        'WaterS', 'Prcp', 'VPD', 'Temp'
    ]
    info_df = pd.DataFrame(columns=['VarSou', 'VarTar'])
    for y_var in y_vars:
        for x_var in x_vars:
            if y_var == x_var:
                continue
            if y_var == 'Temp' and x_var == 'Fert':
                continue
            else:
                Direct_Strength = df_edge_direct[
                    (df_edge_direct['VarSou'] == y_var) &
                    (df_edge_direct['VarTar'] == x_var)]['Strength'].values[0]
                Direct_Strength_Edges_Count = df_edge_direct[
                    (df_edge_direct['VarSou'] == y_var)
                    & (df_edge_direct['VarTar'] == x_var)]['Count'].values[0]
                Indirect_Strength = df_edge_indirect[
                    (df_edge_indirect['VarSou'] == y_var)
                    & (df_edge_indirect['VarTar'] == x_var
                       )]['Indirect_Strength_Mean'].values[0]
                Indirect_Strength_Paths_Count = df_edge_indirect[
                    (df_edge_indirect['VarSou'] == y_var)
                    & (df_edge_indirect['VarTar'] == x_var
                       )]['Paths_Count'].values[0]
                info_df = info_df.append(pd.DataFrame({
                    'VarSou': [y_var],
                    'VarTar': [x_var],
                    'Direct_Strength': [Direct_Strength],
                    'Direct_Strength_Edges_Count':
                    [Direct_Strength_Edges_Count],
                    'Indirect_Strength': [Indirect_Strength],
                    'Indirect_Strength_Paths_Count':
                    [Indirect_Strength_Paths_Count],
                    'Total_Strength': [Direct_Strength + Indirect_Strength],
                }),
                                         ignore_index=True)

    fig = plt.figure(figsize=(14, 8))

    rect1 = [0.1, 0.1, 0.2, 0.85]  # [左, 下, 宽, 高] 规定的矩形区域 （全部是0~1之间的数，表示比例）
    rect2 = [0.36, 0.1, 0.2, 0.85]
    rect3 = [0.62, 0.1, 0.25, 0.85]

    ax1 = plt.axes(rect1)

    # ax1 = fig.add_subplot(1, 3, 1)

    x_map_order = dict(zip(x_vars, range(len(x_vars))))
    y_map_order = dict(zip(y_vars, range(len(y_vars))))
    info_df['x_order'] = info_df['VarTar'].map(x_map_order)
    info_df['y_order'] = info_df['VarSou'].map(y_map_order)

    # edge_color = []
    # for u, v, d in sub_net.edges(data=True):
    #     strength = float(d['strength'])
    #     if strength < 0:
    #         edge_color.append(cm.Blues(-strength))
    #     else:
    #         edge_color.append(cm.Reds(strength))

    # cm0 = plt.get_cmap('bwr')
    # cNorm0 = matplotlib.colors.Normalize(vmin=-0.5, vmax=0.5)

    # cm0 = ListedColormap([
    #     '#083A79',
    #     '#2473B6',
    #     '#69ACD6',
    #     '#C0D9ED',
    #     '#ECF3FB',
    #     '#FFECE5',
    #     '#FCB599',
    #     '#FA6648',
    #     '#CD1A1E',
    #     '#78040F',
    # ])

    cm1 = ListedColormap([
        '#083A79',
        '#2473B6',
        '#69ACD6',
        '#C0D9ED',
        '#FCB599',
        '#FA6648',
        '#CD1A1E',
        '#78040F',
    ])

    cNorm1 = BoundaryNorm([-0.8, -0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.8],
                          cm1.N)
    axcolorbar = ax1.scatter(x=info_df['x_order'],
                             y=info_df['y_order'],
                             s=info_df['Direct_Strength_Edges_Count'] * 15,
                             edgecolors='black',
                             linewidths=0.2,
                             c=info_df['Direct_Strength'],
                             norm=cNorm1,
                             cmap=cm1,
                             label='Direct Strength')

    ax1.set_xlim(-0.5, 3 - 0.5)

    ax1.set_xticks([0, 1, 2])
    ax1.set_xticklabels(x_vars, fontsize=12, fontfamily='Arial')

    ax1.set_yticks(range(0, 11))
    ax1.set_yticklabels(y_vars, fontsize=12, fontfamily='Arial')
    ax1.set_xlabel('Investment Factors',
                   fontsize=14,
                   fontfamily='Arial')
    ax1.set_ylabel('Factors', fontsize=14, fontfamily='Arial')

    # cb = fig.colorbar(axcolorbar, ax=ax1, orientation='vertical')
    # cb.set_ticks([-0.6, -0.45, -0.3, -0.15, 0, 0.15, 0.3, 0.45, 0.6])
    # cb.set_ticklabels([-0.6, -0.45, -0.3, -0.15, 0, 0.15, 0.3, 0.45, 0.6])
    # for l in cb.ax.xaxis.get_ticklabels():
    #     l.set_family('Arial')
    #     l.set_size(14)
    # cb.set_label('Strength of edge', fontsize=14, fontfamily='Arial')
    ax1.set_title(
        'Direct Strength',
        weight='bold',
        #   size='medium',
        position=(0.5, 1.2),
        horizontalalignment='center',
        #   verticalalignment='center'
    )

    #--------------------------------------------------------------
    ax2 = plt.axes(rect2)
    # ax2 = fig.add_subplot(1, 3, 2)
    cm1 = ListedColormap([
        '#083A79',
        '#2473B6',
        '#69ACD6',
        '#C0D9ED',
        '#FCB599',
        '#FA6648',
        '#CD1A1E',
        '#78040F',
    ])

    cNorm1 = BoundaryNorm([-0.8, -0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.8],
                          cm1.N)

    x_map_order = dict(zip(x_vars, range(len(x_vars))))
    y_map_order = dict(zip(y_vars, range(len(y_vars))))
    info_df['x_order'] = info_df['VarTar'].map(x_map_order)
    info_df['y_order'] = info_df['VarSou'].map(y_map_order)

    axcolorbar = ax2.scatter(x=info_df['x_order'],
                             y=info_df['y_order'],
                             s=info_df['Indirect_Strength_Paths_Count'] / 2,
                             edgecolors='black',
                             linewidths=0.2,
                             c=info_df['Indirect_Strength'],
                             norm=cNorm1,
                             cmap=cm1,
                             label='Indirect Strength')

    ax2.set_xlim(-0.5, 3 - 0.5)

    ax2.set_xticks([0, 1, 2])
    ax2.set_xticklabels(x_vars, fontsize=12, fontfamily='Arial')

    ax2.set_yticks(range(0, 11))
    ax2.set_yticklabels(y_vars, fontsize=12, fontfamily='Arial')
    ax2.set_xlabel('Investment Factors',
                   fontsize=14,
                   fontfamily='Arial')
    # ax2.set_ylabel('Other vars', fontsize=12, fontfamily='Arial')

    # cb = fig.colorbar(axcolorbar, ax=ax2, orientation='vertical')
    # cb.set_ticks([-0.6, -0.45, -0.3, -0.15, 0, 0.15, 0.3, 0.45, 0.6])
    # cb.set_ticklabels([-0.6, -0.45, -0.3, -0.15, 0, 0.15, 0.3, 0.45, 0.6])
    # for l in cb.ax.xaxis.get_ticklabels():
    #     l.set_family('Arial')
    #     l.set_size(14)
    # # cb.set_label('Strength of edge', fontsize=14, fontfamily='Arial')
    ax2.set_title('Indirect Strength',
                  weight='bold',
                  position=(0.5, 1.2),
                  horizontalalignment='center')

    #-------------------------------------------
    ax3 = plt.axes(rect3)
    # ax3 = fig.add_subplot(1, 3, 3)
    cm1 = ListedColormap([
        '#083A79',
        '#2473B6',
        '#69ACD6',
        '#C0D9ED',
        '#FCB599',
        '#FA6648',
        '#CD1A1E',
        '#78040F',
    ])

    cNorm1 = BoundaryNorm([-0.8, -0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.8],
                          cm1.N)

    x_map_order = dict(zip(x_vars, range(len(x_vars))))
    y_map_order = dict(zip(y_vars, range(len(y_vars))))
    info_df['x_order'] = info_df['VarTar'].map(x_map_order)
    info_df['y_order'] = info_df['VarSou'].map(y_map_order)

    info_df['Direct_Strength_abs'] = abs(info_df['Direct_Strength'])
    info_df['Indirect_Strength_abs'] = abs(info_df['Indirect_Strength'])
    info_df['Total_Strength_abs'] = info_df['Direct_Strength_abs'] + info_df[
        'Indirect_Strength_abs']

    info_df['Direct_Strength_per'] = info_df['Direct_Strength_abs'] / info_df[
        'Total_Strength_abs']
    info_df['Indirect_Strength_per'] = info_df[
        'Indirect_Strength_abs'] / info_df['Total_Strength_abs']

    for y_var in y_vars:
        for x_var in x_vars:
            if y_var == x_var:
                continue
            if y_var == 'Temp' and x_var == 'Fert':
                continue
            else:
                ac = info_df[(info_df['VarSou'] == y_var) & (
                    info_df['VarTar'] == x_var)]['Direct_Strength'].values[0]
                bc = info_df[(info_df['VarSou'] == y_var) & (
                    info_df['VarTar'] == x_var)]['Indirect_Strength'].values[0]

                ao = info_df[(info_df['VarSou'] == y_var) &
                             (info_df['VarTar'] == x_var)]['x_order'].values[0]
                bo = info_df[(info_df['VarSou'] == y_var) &
                             (info_df['VarTar'] == x_var)]['y_order'].values[0]
                ar = info_df[(info_df['VarSou'] == y_var)
                             & (info_df['VarTar'] == x_var
                                )]['Direct_Strength_per'].values[0]
                br = info_df[(info_df['VarSou'] == y_var)
                             & (info_df['VarTar'] == x_var
                                )]['Indirect_Strength_per'].values[0]
                ss = info_df[
                    (info_df['VarSou'] == y_var)
                    & (info_df['VarTar'] == x_var)]['Total_Strength'].values[0]

                drawPieMarker(p_ax=ax3,
                              xs=ao,
                              ys=bo,
                              ratios=[ar, br],
                              size=abs(ss) * 1000,
                              colors=[ac, bc],
                              norm=cNorm1,
                              cmap=cm1)

    # axcolorbar = ax3.scatter(x=info_df['x_order'],
    #                          y=info_df['y_order'],
    #                          s=60,
    #                          edgecolors='black',
    #                          linewidths=0.2,
    #                          c=info_df['Total_Strength'],
    #                          norm=cNorm3,
    #                          cmap=cm3,
    #                          label='Total Strength')

    ax3.set_xlim(-0.5, 3 - 0.5)

    ax3.set_xticks([0, 1, 2])
    ax3.set_xticklabels(x_vars, fontsize=12, fontfamily='Arial')

    ax3.set_yticks(range(0, 11))
    ax3.set_yticklabels(y_vars, fontsize=12, fontfamily='Arial')
    ax3.set_xlabel('Investment Factors',
                   fontsize=14,
                   fontfamily='Arial')
    # ax2.set_ylabel('Other vars', fontsize=12, fontfamily='Arial')

    cb = fig.colorbar(axcolorbar, ax=ax3, orientation='vertical')
    cb.set_ticks([-0.8, -0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.8])
    cb.set_ticklabels([-0.8, -0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.8])
    for l in cb.ax.xaxis.get_ticklabels():
        l.set_family('Arial')
        l.set_size(12)
    # cb.set_label('Strength of edge', fontsize=14, fontfamily='Arial')
    ax3.set_title('Total Strength',
                  weight='bold',
                  position=(0.5, 1.2),
                  horizontalalignment='center')

    plt.savefig(OUT_PATH + 'Figures//Show_all_in_one_LinkType_Investment.pdf')


def drawPieMarker(p_ax, xs, ys, ratios, size, colors, norm, cmap):
    assert sum(ratios) <= 1.1, 'sum of ratios needs to be < 1'
    markers = []
    previous = 0
    # calculate the points of the pie pieces
    for color, ratio in zip(colors, ratios):
        this = 2 * np.pi * ratio + previous
        x = [0] + np.cos(np.linspace(previous, this, 30)).tolist() + [0]
        y = [0] + np.sin(np.linspace(previous, this, 30)).tolist() + [0]
        xy = np.column_stack([x, y])
        previous = this
        markers.append({
            'marker': xy,
            # 's': np.abs(xy).max()**2 * np.array(sizes),
            's': size,
            # 'facecolor': color,
            'c': color,
            'norm': norm,
            'cmap': cmap
        })

    # scatter each of the pie pieces to create pies
    for marker in markers:
        p_ax.scatter(xs, ys, **marker)


if __name__ == "__main__":
    main()