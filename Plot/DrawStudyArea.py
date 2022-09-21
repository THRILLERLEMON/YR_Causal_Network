# -*- coding: utf-8 -*-
# @Time    : 2019/12/5 15:20
# @Author  : THRILLER柠檬
# @Email   : thrillerlemon@outlook.com
# @File    : BaseMapTest.py
# @Software: PyCharm
# Backup from GeoAgent_GlobalClimate
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.collections import PolyCollection
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
from matplotlib.patches import PathPatch

outPutPath = 'C:\\Users\\Neverland\\Desktop\\'


def main():
    # ******Main******
    print(time.strftime('%H:%M:%S', time.localtime(time.time())))

    fig = plt.figure(figsize=(20, 15))
    ax = fig.add_subplot()
    map = Basemap(
        projection='nsper',
        lat_0=35,
        lon_0=110,
        resolution='l',
        satellite_height=2800000.,
    )
    # Now draw the map
    # map.drawcountries(linewidth=0.5)
    # map.drawstates()
    map.drawcoastlines(linewidth=0.2, color='#666666')
    # fill in color
    # map.fillcontinents(color='gray', lake_color='#7396FE')
    # map.drawmapboundary(fill_color='#7396FE', color='#404243', linewidth=0.5)
    # draw lon and lat
    map.drawparallels(np.arange(-90., 91., 15.),zorder=1)
    map.drawmeridians(np.arange(-180., 181., 15.),zorder=1)
    # map.bluemarble(scale=0.5, alpha=0.8)
    map.shadedrelief(alpha=0.6)
    map.etopo(scale=0.5, alpha=0.3)
    # map.drawrivers(color='#30c1e5')

    map.readshapefile(
        'D:\\YR_SDG_Network-Data\\Figure\\StudyArea\\GA_WGS84_YR_dissolve',
        'GA_WGS84_YR_dissolve',
        color='#73ffdf',
        linewidth=1,
        drawbounds=False)
    patches = []
    for info, shape in zip(map.GA_WGS84_YR_dissolve, map.GA_WGS84_YR_dissolve):
        patches.append(Polygon(np.array(shape), True))

    ax.add_collection(
        PatchCollection(patches,
                        facecolor='#f1801f',
                        edgecolor='#ffffff',
                        linewidths=0.5,
                        zorder=2,alpha=0.8))

    map.readshapefile('E:\\GIS_DATA\\world_rivers\\world_rivers',
                      'world_rivers',
                      color='#399dd2',
                      linewidth=1)
    plt.savefig(outPutPath + 'YRonEarth.pdf')


# Run main
if __name__ == "__main__":
    main()
