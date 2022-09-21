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
    x_vars = [
        'CropG', 'FruitG', 'IncomeR', 'Fert', 'AgriM', 'RuralE', 'WaterG',
        'WaterS', 'Prcp', 'VPD', 'Temp','CO2'
    ]
    y_vars = [
        'CropG', 'FruitG', 'IncomeR', 'Fert', 'AgriM', 'RuralE', 'WaterG',
        'WaterS', 'Prcp', 'VPD', 'Temp','CO2'
    ]
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
    info_df = pd.DataFrame(columns=['VarSou', 'VarTar'])
    for y_var in y_vars:
        for x_var in x_vars:
            if y_var == x_var:
                continue
            else:
                try:
                    Direct_Strength = df_edge_direct[
                        (df_edge_direct['VarSou'] == y_var)
                        & (df_edge_direct['VarTar'] == x_var
                           )]['Strength'].values[0]
                    Direct_Strength_Edges_Count = df_edge_direct[
                        (df_edge_direct['VarSou'] == y_var)
                        &
                        (df_edge_direct['VarTar'] == x_var)]['Count'].values[0]
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
                        'Total_Strength':
                        [Direct_Strength + Indirect_Strength],
                    }),
                                             ignore_index=True)
                except:
                    continue
    fig = plt.figure(figsize=(10, 10))
    # # [左, 下, 宽, 高] 规定的矩形区域 （全部是0~1之间的数，表示比例）
    rect3 = [0.2, 0.2, 0.7, 0.7]
    rectcb = [0.91, 0.35, 0.02, 0.55]
    ax3 = plt.axes(rect3)
    # ax3 = fig.add_subplot(111)

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

    axcolorbar = None

    for y_var in y_vars:
        for x_var in x_vars:
            if y_var == x_var:
                continue
            else:
                try:
                    ac = info_df[(info_df['VarSou'] == y_var)
                                 & (info_df['VarTar'] == x_var
                                    )]['Direct_Strength'].values[0]
                    bc = info_df[(info_df['VarSou'] == y_var)
                                 & (info_df['VarTar'] == x_var
                                    )]['Indirect_Strength'].values[0]

                    ao = info_df[(info_df['VarSou'] == y_var) & (
                        info_df['VarTar'] == x_var)]['x_order'].values[0]
                    bo = info_df[(info_df['VarSou'] == y_var) & (
                        info_df['VarTar'] == x_var)]['y_order'].values[0]
                    ar = info_df[(info_df['VarSou'] == y_var)
                                 & (info_df['VarTar'] == x_var
                                    )]['Direct_Strength_per'].values[0]
                    br = info_df[(info_df['VarSou'] == y_var)
                                 & (info_df['VarTar'] == x_var
                                    )]['Indirect_Strength_per'].values[0]
                    ss = info_df[(info_df['VarSou'] == y_var)
                                 & (info_df['VarTar'] == x_var
                                    )]['Total_Strength'].values[0]
                    # usually Direct_Strength_per is bigger,so first fraw Direct_Strength
                    axcolorbar = drawPieMarker(p_ax=ax3,
                                               xs=ao,
                                               ys=bo,
                                               ratios=[ar, br],
                                               size=abs(ss) * 1000,
                                               colors=[ac, bc],
                                               norm=cNorm1,
                                               cmap=cm1)
                except:
                    continue
    # axcolorbar = ax3.scatter(x=info_df['x_order'],
    #                          y=info_df['y_order'],
    #                          s=60,
    #                          edgecolors='black',
    #                          linewidths=0.2,
    #                          c=info_df['Total_Strength'],
    #                          norm=cNorm3,
    #                          cmap=cm3,
    #                          label='Total Strength')

    ax3.set_xlim(-0.5, 12 - 0.5)

    ax3.set_xticks(range(0, 12))
    x_lables = [VAR_LABEL_DICT[v] for v in x_vars]
    ax3.set_xticklabels(x_lables, fontsize=12, fontfamily='Arial')

    ax3.set_yticks(range(0, 12))
    y_lables = [VAR_LABEL_DICT[v] for v in y_vars]
    ax3.set_yticklabels(y_lables, fontsize=12, fontfamily='Arial')
    ax3.set_xlabel('Effect (influenced)', fontsize=14, fontfamily='Arial')
    ax3.set_ylabel('Cause (influencing)', fontsize=14, fontfamily='Arial')

    cbax = fig.add_axes(rectcb)
    cb = fig.colorbar(axcolorbar, cax=cbax, orientation='vertical')
    # cb = fig.colorbar(axcolorbar, ax=ax3, orientation='vertical')
    cb.set_ticks([-0.8, -0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.8])
    cb.set_ticklabels([-0.8, -0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.8])
    for l in cb.ax.xaxis.get_ticklabels():
        l.set_family('Arial')
        l.set_size(12)
    cb.set_label('Strength', fontsize=14, fontfamily='Arial')
    # ax3.set_title('Total Strength',
    #               weight='bold',
    #               position=(0.5, 1.2),
    #               horizontalalignment='center')

    ax3.scatter([15], [3],
                marker='o',
                color=['#827f80'],
                s=[0.5 * 1000],
                alpha=0.8,
                label='±0.5',
                edgecolors='#595c5d',
                linewidths=1,
                zorder=100)
    ax3.scatter([15], [2],
                marker='o',
                color=['#827f80'],
                s=[0.3 * 1000],
                alpha=0.8,
                label='±0.3',
                edgecolors='#595c5d',
                linewidths=1,
                zorder=100)
    ax3.scatter([15], [1],
                marker='o',
                color=['#827f80'],
                s=[0.1 * 1000],
                alpha=0.8,
                label='±0.1',
                edgecolors='#595c5d',
                linewidths=1,
                zorder=100)

    ax3.legend(bbox_to_anchor=(1.13, 0.18),
               ncol=1,
               prop={
                   'family': 'Arial',
                   'size': 12,
               },
               labelspacing=1,
               borderpad=0.5,
               handletextpad=0.1,
               markerscale=1,
               frameon=False)

    plt.savefig(OUT_PATH + 'Figures//Show_all_in_one_LinkType_whole_net.pdf')


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
    axcolorbar = None
    # scatter each of the pie pieces to create pies
    for marker in markers:
        axcolorbar = p_ax.scatter(xs, ys, **marker)
    return axcolorbar


if __name__ == "__main__":
    main()