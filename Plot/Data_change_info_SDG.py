from ast import Num
from cProfile import label
import numpy as np
import pandas as pd
from scipy.stats import linregress
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator, yscale
from sklearn.linear_model import LinearRegression

VARS_LIST_Climate = ['CO2', 'Temp', 'VPD']
VARS_LIST_Water = ['Prcp', 'WaterS', 'WaterG']
VARS_LIST_Investment = ['RuralE', 'AgriM', 'Fert']
VARS_LIST_Harvest = ['CropG', 'FruitG', 'IncomeR']

VARS_LIST = VARS_LIST_Climate + VARS_LIST_Water + VARS_LIST_Investment + VARS_LIST_Harvest
# VARS_LIST = VARS_LIST_Investment + VARS_LIST_Harvest
datapath = 'D:\\YR_SDG_Network\\data\\Data_for_SDG\\'
# datapath = 'D:\\YR_SDG_Network\\data\\Data_for_SDG_years_net\\'
geopath = 'D:\\YR_SDG_Network\\data\\GeoAgent\\'
out_info = pd.DataFrame(columns=[VARS_LIST])
for var in VARS_LIST:
    # yearly data
    var_data = pd.read_csv(datapath + 'YR_GA_Info_' + var + '_yearly' + '.csv',
                           index_col='GA_ID')
    if var == 'WaterS':
        var_data = pd.read_csv(
            'D:\\YR_SDG_Network-Data\\Figure\\DataInfo\\YR_GA_Info_WaterS_yearly.csv',
            index_col='GA_ID')

    var_data_filled = var_data.dropna(how='all')
    var_data_filled = var_data_filled.fillna(method='ffill', axis=1)
    var_data_filled = var_data_filled.fillna(method='bfill', axis=1)
    var_data_filled.loc['mean'] = var_data_filled.apply(lambda x: x.mean())

    firstyear = int(var_data_filled.columns[0])
    lastyear = int(var_data_filled.columns[len(var_data_filled.columns) - 1])
    fig = plt.figure(figsize=(5, 2), dpi=300)
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    ax = fig.add_subplot(111)
    x_data = np.linspace(firstyear, lastyear, len(var_data_filled.columns))
    y_data = var_data_filled.loc['mean']

    y_scale = 1
    y_label = ''
    x_ticks = [2005, 2010, 2015]
    if var == 'CO2':
        y_scale = 1
        y_label = 'Unit:ppm'
        x_ticks = [1980, 1990, 2000, 2010, 2020]
    if var == 'Temp':
        y_data = y_data - 273.15
        y_scale = 1
        y_label = 'Unit:°C'
        x_ticks = [1981, 1990, 2000, 2010, 2020]
    if var == 'VPD':
        y_scale = 1
        y_label = 'Unit:$\mathregular{10^{-2}kPa}$'
        x_ticks = [1981, 1990, 2000, 2010, 2020]
    if var == 'Prcp':
        y_scale = 1
        y_label = 'Unit:m'
        x_ticks = [1981, 1990, 2000, 2010, 2020]
    if var == 'WaterS':
        y_scale = 0.000001
        y_label = 'Unit:$\mathregular{km^2}$'
        x_ticks = [1985, 1990, 2000, 2010, 2020]
    if var == 'WaterG':
        y_scale = 1
        y_label = 'Unit:cm'
        x_ticks = [2002, 2007, 2012, 2016]
    if var == 'RuralE':
        y_scale = 0.01
        y_label = 'Unit:$\mathregular{10^6kWh}$'
        x_ticks = [1980, 1990, 2000, 2010, 2018]
    if var == 'AgriM':
        y_scale = 1
        y_label = 'Unit:$\mathregular{10^4kW}$'
        x_ticks = [1980, 1990, 2000, 2010, 2018]
    if var == 'Fert':
        y_scale = 0.0001
        y_label = 'Unit:$\mathregular{10^4t}$'
        x_ticks = [1980, 1990, 2000, 2010, 2018]
    if var == 'CropG':
        y_scale = 0.0001
        y_label = 'Unit:$\mathregular{10^4t}$'
        x_ticks = [1980, 1990, 2000, 2010, 2018]
    if var == 'FruitG':
        y_scale = 0.0001
        y_label = 'Unit:$\mathregular{10^4t}$'
        x_ticks = [1980, 1990, 2000, 2010, 2018]
    if var == 'IncomeR':
        y_scale = 0.01
        y_label = 'Unit:$\mathregular{10^2yuan}$'
        x_ticks = [1980, 1990, 2000, 2010, 2018]

    plt.plot(x_data,
             y_data * y_scale,
             label=y_label,
             color='k',
             linewidth=2,
             linestyle='-',
             marker='o',
             ms=3.5)

    model = LinearRegression()
    X = np.array(x_data).reshape(-1, 1)
    Y = np.array(y_data * y_scale).reshape(-1, 1)
    model.fit(X, Y)
    X2 = [[firstyear], [firstyear + 2], [firstyear + 4], [lastyear]]
    y2 = model.predict(X2)
    plt.plot(
        X2,
        y2,
        color='tomato',
        linewidth=2.5,
        linestyle='--',
        #  label='Trend line'
    )

    plt.legend(
        # loc='upper left',
        ncol=1,
        prop={
            'size': 16,
        },
        labelspacing=1,
        borderpad=0.4,
        handletextpad=0,
        markerscale=0,
        handlelength=0,
        frameon=True)

    plt.xlim(firstyear - 0.5, lastyear + 0.5)
    ax.tick_params(pad=2.3, width=2, length=5.3)
    plt.yticks(fontproperties='Times New Roman', size=18)
    plt.xticks(x_ticks, fontproperties='Times New Roman', size=18)
    plt.xlabel('Year', fontproperties='Times New Roman', size=16, labelpad=1)
    # plt.ylabel(y_label, fontproperties='Times New Roman', size=16)
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    plt.savefig('D:\\YR_SDG_Network-Data\\Figure\\DataInfo\\change_of_' + var +
                '.svg',
                bbox_inches='tight',
                transparent=True)

    centroid_data = pd.read_csv(geopath + 'GA_Centroid.csv')
    for agent in list(centroid_data['GA_ID']):
        y_ori = var_data.loc[agent]
        first_idx = y_ori.first_valid_index()
        last_idx = y_ori.last_valid_index()
        y_short = y_ori.loc[first_idx:last_idx]
        y = y_short.fillna(method='ffill')
        x = np.arange(len(y))
        slope, intercept, r, p, std_err = linregress(x, y)
        if p < 0.05:
            out_info.loc[agent, var+'_S'] = slope
    out_info.to_csv('D:\\YR_SDG_Network-Data\\Figure\\DataInfo\\DataChangeInfo.csv')