import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, RegularPolygon
from matplotlib.path import Path
from matplotlib.projections.polar import PolarAxes
from matplotlib.projections import register_projection
from matplotlib.spines import Spine
from matplotlib.transforms import Affine2D
from ast import Num
from cProfile import label
import numpy as np
import pandas as pd
from scipy.stats import linregress
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator, yscale
from sklearn.linear_model import LinearRegression

plt.rc('font', family='Arial')


def radar_factory(num_vars, frame='circle'):
    """
    Create a radar chart with `num_vars` axes.

    This function creates a RadarAxes projection and registers it.

    Parameters
    ----------
    num_vars : int
        Number of variables for radar chart.
    frame : {'circle', 'polygon'}
        Shape of frame surrounding axes.

    """
    # calculate evenly-spaced axis angles
    theta = np.linspace(0, 2 * np.pi, num_vars, endpoint=False)

    class RadarAxes(PolarAxes):

        name = 'radar'
        # use 1 line segment to connect specified points
        RESOLUTION = 1

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            # rotate plot such that the first axis is at the top
            self.set_theta_zero_location('N')

        def fill(self, *args, closed=True, **kwargs):
            """Override fill so that line is closed by default"""
            return super().fill(closed=closed, *args, **kwargs)

        def plot(self, *args, **kwargs):
            """Override plot so that line is closed by default"""
            lines = super().plot(*args, **kwargs)
            for line in lines:
                self._close_line(line)

        def _close_line(self, line):
            x, y = line.get_data()
            # FIXME: markers at x[0], y[0] get doubled-up
            if x[0] != x[-1]:
                x = np.append(x, x[0])
                y = np.append(y, y[0])
                line.set_data(x, y)

        def set_varlabels(self, labels):
            self.set_thetagrids(np.degrees(theta), labels, family='Arial')

        def _gen_axes_patch(self):
            # The Axes patch must be centered at (0.5, 0.5) and of radius 0.5
            # in axes coordinates.
            if frame == 'circle':
                return Circle((0.5, 0.5), 0.5)
            elif frame == 'polygon':
                return RegularPolygon((0.5, 0.5),
                                      num_vars,
                                      radius=.5,
                                      edgecolor="k")
            else:
                raise ValueError("Unknown value for 'frame': %s" % frame)

        def _gen_axes_spines(self):
            if frame == 'circle':
                return super()._gen_axes_spines()
            elif frame == 'polygon':
                # spine_type must be 'left'/'right'/'top'/'bottom'/'circle'.
                spine = Spine(axes=self,
                              spine_type='circle',
                              path=Path.unit_regular_polygon(num_vars))
                # unit_regular_polygon gives a polygon of radius 1 centered at
                # (0, 0) but we want a polygon of radius 0.5 centered at (0.5,
                # 0.5) in axes coordinates.
                spine.set_transform(Affine2D().scale(.5).translate(.5, .5) +
                                    self.transAxes)
                return {'polar': spine}
            else:
                raise ValueError("Unknown value for 'frame': %s" % frame)

    register_projection(RadarAxes)
    return theta


def get_mean_slope_data():
    VARS_LIST_Climate = ['CO2', 'Temp', 'VPD']
    VARS_LIST_Water = ['Prcp', 'WaterS', 'WaterG']
    VARS_LIST_Investment = ['RuralE', 'AgriM', 'Fert']
    VARS_LIST_Harvest = ['CropG', 'FruitG', 'IncomeR']

    VARS_LIST = VARS_LIST_Climate + VARS_LIST_Water + VARS_LIST_Investment + VARS_LIST_Harvest
    datapath = 'D:\\YR_SDG_Network\\data\\Data_for_SDG\\'
    geopath = 'D:\\YR_SDG_Network\\data\\GeoAgent\\'
    vars_m = [*map(lambda x: x + '_M', VARS_LIST)]
    vars_s = [*map(lambda x: x + '_S', VARS_LIST)]
    vars_s_per = [*map(lambda x: x + '_S_per', VARS_LIST)]
    out_info = pd.DataFrame(columns=[vars_m + vars_s])
    for var in VARS_LIST:
        # yearly data
        var_data = pd.read_csv(datapath + 'YR_GA_Info_' + var + '_yearly' +
                               '.csv',
                               index_col='GA_ID')
        if var == 'WaterS':
            var_data = pd.read_csv(
                'D:\\YR_SDG_Network-Data\\Figure\\DataInfo\\YR_GA_Info_WaterS_yearly.csv',
                index_col='GA_ID')
        centroid_data = pd.read_csv(geopath + 'GA_Centroid.csv')
        for agent in list(centroid_data['GA_ID']):
            y_ori = var_data.loc[agent]
            first_idx = y_ori.first_valid_index()
            last_idx = y_ori.last_valid_index()
            y_short = y_ori.loc[first_idx:last_idx]
            y = y_short.fillna(method='ffill')
            x = np.arange(len(y))
            slope, intercept, r, p, std_err = linregress(x, y)
            if first_idx:
                first_num = int(first_idx) * slope + intercept
                last_num = int(last_idx) * slope + intercept
                change_ratio = 100 * (last_num - first_num) / first_num
                out_info.loc[agent, var + '_S_per'] = change_ratio
            out_info.loc[agent, var + '_S'] = slope
            out_info.loc[agent, var + '_M'] = y.mean()
    out_info.to_csv('D:\\YR_SDG_Network\\data\\GeoAgent\\DataInfo.csv')
    return out_info


def example_data():
    group_data = pd.read_csv(
        'D:\\YR_SDG_Network\\data\\GeoAgent\\grouping_202206\\Group_5.csv',
        index_col='GA_ID')
    grouped_df = group_data.groupby('SS_GROUP')
    grouped_df_mean = grouped_df.mean().reset_index()
    var_list = [
        'WaterS', 'WaterG', 'RuralE', 'AgriM', 'Fert', 'CropG', 'FruitG',
        'IncomeR'
    ]
    Gs_m = []
    Gs_s = []
    for i in [1, 2, 3, 4, 5]:
        group_m = []
        group_s = []
        for var in var_list:
            mmm = grouped_df_mean[grouped_df_mean['SS_GROUP'] == i]
            m_value = mmm[var + '_M']
            s_value = mmm[var + '_S']
            if var == 'WaterS':
                m_value = m_value / 1000000
                s_value = s_value / 10000
            if var == 'WaterG':
                m_value = m_value * -10
                s_value = s_value * -100
            if var == 'RuralE':
                m_value = m_value / 100
                s_value = s_value / 10
            if var == 'AgriM':
                m_value = m_value
                s_value = s_value * 100
            if var == 'Fert':
                m_value = m_value / 1000
                s_value = s_value / 10
            if var == 'CropG':
                m_value = m_value / 10000
                s_value = s_value / 100
            if var == 'FruitG':
                m_value = m_value / 1000
                s_value = s_value / 100
            if var == 'IncomeR':
                m_value = m_value / 100
                s_value = s_value / 10
            group_m.append(m_value)
            group_s.append(s_value)
        Gs_m.append(group_m)
        Gs_s.append(group_s)

    data = [
        var_list, ('Average in different modes', Gs_m),
        ('Change ratio in different modes', Gs_s)
    ]
    #     [
    #         'CropG', 'FruitG', 'AgriM', 'Fert', 'WaterS', 'WaterG', '50', '50',
    #         '50'
    #     ],
    #     ('Average in different patterns',
    #      [[2.1599, 0.26969, 5.439, 3.695, 52.076481, 4.832, 50, 50, 50],
    #       [11.6838, 41.68054, 16.837, 11.389, 8.997374, 20.792, 50, 50, 50],
    #       [9.4772, 22.4375, 16.466, 7.732, 7.451184, 30.968, 50, 50, 50],
    #       [29.2845, 86.75342, 39.287, 30.606, 19.954703, 29.984, 50, 50, 50]]),
    #     ('Change ratio in different patterns',
    #      [[0.9376, 0.03113, 21.43, 38.8, 14.1076, 1.141, 50, 50, 50],
    #       [28.5623, 31.98302, 67.96, 42.8, 35.1538, 49.61, 50, 50, 50],
    #       [25.28, 17.99746, 68.1, 23.8, 21.7932, 84.26, 50, 50, 50],
    #       [26.66, 61.82179, 86.15, 5.6, 61.4148, 59.52, 50, 50, 50]])
    # ]
    return data


if __name__ == '__main__':
    # get_mean_slope_data()
    data = example_data()
    N = 8
    theta = radar_factory(N, frame='polygon')
    spoke_labels = data.pop(0)

    fig, axs = plt.subplots(figsize=(12, 8),
                            nrows=1,
                            ncols=2,
                            subplot_kw=dict(projection='radar'))
    fig.subplots_adjust(wspace=0.25, hspace=0.25, top=0.8, bottom=0.2)

    colors = ['#006014', '#7aa92c', '#fffc47', '#ff9729', '#ff200e']
    # Plot the four cases from the example data on separate axes
    for ax, (title, case_data) in zip(axs.flat, data):
        ax.set_rgrids([20, 40, 60, 80, 100, 120])
        ax.set_title(title,
                     weight='bold',
                     size=14,
                     family='Arial',
                     position=(0.5, 0.5),
                     horizontalalignment='center',
                     verticalalignment='center')
        for d, color in zip(case_data, colors):
            ax.plot(theta, d, color=color)
            ax.fill(theta, d, facecolor=color, alpha=0.25)
        ax.set_varlabels(spoke_labels)

    # add legend relative to top-left plot
    labels = ('Mode 1', 'Mode 2', 'Mode 3', 'Mode 4', 'Mode 5')
    legend = axs[0].legend(
        labels,
        loc=(0.98, 0),
        labelspacing=0.1,
        prop={
            'family': 'Arial',
            'size': 10,
        },
    )

    fig.text(
        0.1,
        0.18,
        'Average value units: WaterS($\mathregular{km^2}$), WaterG(-mm), RuralE($\mathregular{10^6kWh}$), AgriM($\mathregular{10^4kw}$), Fert($\mathregular{10^3t}$), CropG($\mathregular{10^4t}$), FruitG($\mathregular{10^3t}$), IncomeR($\mathregular{10^2yuan}$)',
        horizontalalignment='left',
        color='black',
        # weight='bold',
        family='Arial',
        size=10)
    fig.text(
        0.1,
        0.15,
        'Change ratio value units: WaterS($\mathregular{10^4m^2/y}$), WaterG(-10mm/y), RuralE($\mathregular{10^5kWh/y}$), AgriM($\mathregular{10^2kw/y}$), Fert(10t/y), CropG($\mathregular{10^2t/y}$), FruitG($\mathregular{10^2t/y}$), IncomeR($\mathregular{*10yuan/y}$)',
        horizontalalignment='left',
        color='black',
        # weight='bold',
        family='Arial',
        size=10)

    # plt.show()
    OUT_PATH = 'D://YR_SDG_Network-Data//Output//'
    if not os.path.exists(OUT_PATH + 'Figures'):
        os.mkdir(OUT_PATH + 'Figures')
    plt.savefig(OUT_PATH + 'Figures//Show_group_info_Radar_chart.pdf')