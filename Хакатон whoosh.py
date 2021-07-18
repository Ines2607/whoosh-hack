#!/usr/bin/env python
# coding: utf-8

# In[1]:


from h3 import h3

import pandas as pd

from shapely.geometry import Point,Polygon

from shapely.geometry import shape

import folium

from cartoframes.viz import Map, Layer, color_continuous_style, palettes, histogram_widget

import numpy as np
import geopandas as gpd

from shapely.geometry import shape, GeometryCollection

import time

import seaborn as sn

from sklearn.cluster import DBSCAN, KMeans

from matplotlib import pyplot as plt


# In[2]:


import osmnx as ox
from shapely.geometry import Point
import geopandas as gpd
import overpy
from shapely.geometry import shape
from shapely.geometry import Polygon
import folium
from folium import Choropleth, Circle, Marker, CircleMarker
from folium.plugins import HeatMap, MarkerCluster

import osmnx as ox

import networkx as nx
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

import matplotlib as mpl
import matplotlib.cm as cm

get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.colors as mcolors
import warnings
warnings.simplefilter("ignore")
from sklearn.linear_model import LinearRegression as LR
import contextily as ctx

from sklearn.metrics import r2_score

import shapely.wkt

import  seaborn as sns


# In[ ]:


#Ранжирование Москвы с точки зрения спроса и безопасности


# In[ ]:


##Что такое безопасность :


# In[5]:


df_infra=gpd.read_file('/Users/intra26/Downloads/moscow_insfrastructure.geojson')


# In[7]:


df_infra.Caption.unique()


# In[8]:


gdf_ws = gpd.read_file('/Users/intra26/Downloads/streets_ws.geojson')


# In[9]:


gdf_ws.head()


# In[20]:


df_sim_crash = pd.read_csv(open(r'/Users/intra26/Downloads/СИМ/dtp_2021.csv'),sep=',')


# In[23]:


df_sim_crash.rename(columns={'adres':'address'}, inplace=True)


# In[19]:


df_sim_crash1 = pd.read_csv(open(r'/Users/intra26/Downloads/СИМ/dtp_2020.csv'),sep=',')


# In[24]:


df_sim_crash1.rename(columns={'adress':'address'}, inplace=True)


# In[26]:


df_sim_crash1.rename(columns={'type_area':'area_type'}, inplace=True)


# In[28]:


df_sim_crash = pd.concat([df_sim_crash ,df_sim_crash1], axis=0)


# In[29]:


df_sim_crash['date']= pd.to_datetime(df_sim_crash['date'])


# In[30]:



def get_gdf(df, geometry, crs):
    return gpd.GeoDataFrame(df, geometry=geometry, crs='epsg:%s'%str(crs))

def get_points(lat, long):
    return[ Point (j,i) for i, j in zip(lat, long)]


# In[32]:


df_sim_crash['geometry'] = get_points(df_sim_crash.lat, df_sim_crash.long)


# In[37]:


gdf_sim_crash=get_gdf(df_sim_crash,'geometry',4326)


# In[50]:


gdf_sim_crash.to_file('gdf_sim_crash.geojson', driver='GeoJSON')


# In[ ]:


# Выделить опасные зоны, коррелирует ли с ws???


# In[38]:


gdf_sim_crash[].plot()


# In[41]:


df_parking = pd.read_csv('/Users/intra26/Downloads/velo_parking.csv')


# In[42]:


df_parking.head()


# In[44]:


df_parking['geometry'] = get_points(df_parking['Latitude_WGS84'], df_parking['Longitude_WGS84'])


# In[45]:


df_parking.geometry


# In[39]:


df_scooter = pd.read_csv('/Users/intra26/Downloads/scooter.csv')


# In[47]:


df_scooter['geometry'] = get_points(df_scooter.lat, df_scooter.lon)


# In[65]:


gdf_scooter=get_gdf(df_scooter,'geometry',4326)


# In[48]:


df_scooter.head()


# In[ ]:


### Модель: надо понять причины аварийности


# In[60]:


def get_h3 (city_boarders):
    
    boarders_gjson= { 
                "type": "Polygon",
                "coordinates": [
                   city_boarders
                ] 
            }

#     boarders = shape( { "type": "Polygon", "crs":"epsg=4326", "coordinates": [city_boarders]})


    hexs = h3.polyfill(boarders_gjson, res=8,geo_json_conformant = True)

    polygonise = lambda hex_id: Polygon(
                            h3.h3_to_geo_boundary(
                            hex_id, geo_json=True)
                                        )

    # %time all_polys = gpd.GeoSeries(list(map(polygonise, hexs)), \
    #                                       index=hexs, \
    #                                       crs="EPSG:4326" \
    #                                      )

    gdf_hex = gpd.GeoDataFrame(list(map(polygonise, hexs)),                                           index=hexs,                                           crs="EPSG:4326",                      columns=['geometry']
                                         )
    fig, ax=plt.subplots(figsize=(13,10))
    gdf_hex.plot(alpha=0.5, linewidth=1, ax=ax)
    ctx.add_basemap(ax=ax, crs='epsg:4326')
    plt.show()
    return gdf_hex


# In[54]:


import h3


# In[52]:


mkad_original=([55.78000432402266,37.84172564285271],[55.775874525970494,37.8381207618713],[55.775626746008065,37.83979446823122],[55.77446586811748,37.84243326983639],[55.771974101091104,37.84262672750849],[55.77114545193181,37.84153238623039],[55.76722010265554,37.841124690460184],[55.76654891107098,37.84239076983644],[55.76258709833121,37.842283558197025],[55.758073999993734,37.8421759312134],[55.75381499999371,37.84198330422974],[55.749277102484484,37.8416827275085],[55.74794544108413,37.84157576190186],[55.74525257875241,37.83897929098507],[55.74404373042019,37.83739676451868],[55.74298009816793,37.838732481460525],[55.743060321833575,37.841183997352545],[55.73938799999373,37.84097476190185],[55.73570799999372,37.84048155819702],[55.73228210777237,37.840095812164286],[55.73080491981639,37.83983814285274],[55.729799917464675,37.83846476321406],[55.72919751082619,37.83835745269769],[55.72859509486539,37.838636380279524],[55.727705075632784,37.8395161005249],[55.722727886185154,37.83897964285276],[55.72034817326636,37.83862557539366],[55.71944437307499,37.83559735744853],[55.71831419154461,37.835370708803126],[55.71765218986692,37.83738169402022],[55.71691750159089,37.83823396494291],[55.71547311301385,37.838056931213345],[55.71221445615604,37.836812846557606],[55.709331054395555,37.83522525396725],[55.70953687463627,37.83269301586908],[55.70903403789297,37.829667367706236],[55.70552351822608,37.83311126588435],[55.70041317726053,37.83058993121339],[55.69883771404813,37.82983872750851],[55.69718947487017,37.82934501586913],[55.69504441658371,37.828926414016685],[55.69287499999378,37.82876530422971],[55.690759754047335,37.82894754100031],[55.68951421135665,37.827697554878185],[55.68965045405069,37.82447346292115],[55.68322046195302,37.83136543914793],[55.67814012759211,37.833554015869154],[55.67295011628339,37.83544184655761],[55.6672498719639,37.837480388885474],[55.66316274139358,37.838960677246064],[55.66046999999383,37.83926093121332],[55.65869897264431,37.839025050262435],[55.65794084879904,37.83670784390257],[55.65694309303843,37.835656529083245],[55.65689306460552,37.83704060449217],[55.65550363526252,37.83696819873806],[55.65487847246661,37.83760389616388],[55.65356745541324,37.83687972750851],[55.65155951234079,37.83515216004943],[55.64979413590619,37.83312418518067],[55.64640836412121,37.82801726983639],[55.64164525405531,37.820614174591],[55.6421883258084,37.818908190475426],[55.64112490388471,37.81717543386075],[55.63916106913107,37.81690987037274],[55.637925371757085,37.815099354492155],[55.633798276884455,37.808769150787356],[55.62873670012244,37.80100123544311],[55.62554336109055,37.79598013491824],[55.62033499605651,37.78634567724606],[55.618768681480326,37.78334147619623],[55.619855533402706,37.77746201055901],[55.61909966711279,37.77527329626457],[55.618770300976294,37.77801986242668],[55.617257701952106,37.778212973541216],[55.61574504433011,37.77784818518065],[55.61148576294007,37.77016867724609],[55.60599579539028,37.760191219573976],[55.60227892751446,37.75338926983641],[55.59920577639331,37.746329965606634],[55.59631430313617,37.73939925396728],[55.5935318803559,37.73273665739439],[55.59350760316188,37.7299954450912],[55.59469840523759,37.7268679946899],[55.59229549697373,37.72626726983634],[55.59081598950582,37.7262673598022],[55.5877595845419,37.71897193121335],[55.58393177431724,37.70871550793456],[55.580917323756644,37.700497489410374],[55.57778089778455,37.69204305026244],[55.57815154690915,37.68544477378839],[55.57472945079756,37.68391050793454],[55.57328235936491,37.678803592590306],[55.57255251445782,37.6743402539673],[55.57216388774464,37.66813862698363],[55.57505691895805,37.617927457672096],[55.5757737568051,37.60443099999999],[55.57749105910326,37.599683515869145],[55.57796291823627,37.59754177842709],[55.57906686095235,37.59625834786988],[55.57746616444403,37.59501783265684],[55.57671634534502,37.593090671936025],[55.577944600233785,37.587018007904],[55.57982895000019,37.578692203704804],[55.58116294118248,37.57327546607398],[55.581550362779,37.57385012109279],[55.5820107079112,37.57399562266922],[55.58226289171689,37.5735356072979],[55.582393529795155,37.57290393054962],[55.581919415056234,37.57037722355653],[55.584471614867844,37.5592298306885],[55.58867650795186,37.54189249206543],[55.59158133551745,37.5297256269836],[55.59443656218868,37.517837865081766],[55.59635625174229,37.51200186508174],[55.59907823904434,37.506808949737554],[55.6062944994944,37.49820432275389],[55.60967103463367,37.494406071441674],[55.61066689753365,37.494760001358024],[55.61220931698269,37.49397137107085],[55.613417718449064,37.49016528606031],[55.61530616333343,37.48773249206542],[55.622640129112334,37.47921386508177],[55.62993723476164,37.470652153442394],[55.6368075123157,37.46273446298218],[55.64068225239439,37.46350692265317],[55.640794546982576,37.46050283203121],[55.64118904154646,37.457627470916734],[55.64690488145138,37.450718034393326],[55.65397824729769,37.44239252645875],[55.66053543155961,37.434587576721185],[55.661693766520735,37.43582144975277],[55.662755031737014,37.43576786245721],[55.664610641628116,37.430982915344174],[55.66778515273695,37.428547447097685],[55.668633314343566,37.42945134592044],[55.66948145750025,37.42859571562949],[55.670813882451405,37.4262836402282],[55.6811141674414,37.418709037048295],[55.68235377885389,37.41922139651101],[55.68359335082235,37.419218771842885],[55.684375235224735,37.417196501327446],[55.68540557585352,37.41607020370478],[55.68686637150793,37.415640857147146],[55.68903015131686,37.414632153442334],[55.690896881757396,37.413344899475064],[55.69264232162232,37.41171432275391],[55.69455101638112,37.40948282275393],[55.69638690385348,37.40703674603271],[55.70451821283731,37.39607169577025],[55.70942491932811,37.38952706878662],[55.71149057784176,37.387778313491815],[55.71419814298992,37.39049275399779],[55.7155489617061,37.385557272491454],[55.71849856042102,37.38388335714726],[55.7292763261685,37.378368238098155],[55.730845879211614,37.37763597123337],[55.73167906388319,37.37890062088197],[55.734703664681774,37.37750451918789],[55.734851959522246,37.375610832015965],[55.74105626086403,37.3723813571472],[55.746115620904355,37.37014935714723],[55.750883999993725,37.36944173016362],[55.76335905525834,37.36975304365541],[55.76432079697595,37.37244070571134],[55.76636979670426,37.3724259757175],[55.76735417953104,37.369922155757884],[55.76823419316575,37.369892695770275],[55.782312184391266,37.370214730163575],[55.78436801120489,37.370493611114505],[55.78596427165359,37.37120164550783],[55.7874378183096,37.37284851456452],[55.7886695054807,37.37608325135799],[55.78947647305964,37.3764587460632],[55.79146512926804,37.37530000265506],[55.79899647809345,37.38235915344241],[55.80113596939471,37.384344043655396],[55.80322699999366,37.38594269577028],[55.804919036911976,37.38711208598329],[55.806610999993666,37.3880239841309],[55.81001864976979,37.38928977249147],[55.81348641242801,37.39038389947512],[55.81983538336746,37.39235781481933],[55.82417822811877,37.393709457672124],[55.82792275755836,37.394685720901464],[55.830447148154136,37.39557615344238],[55.83167107969975,37.39844478226658],[55.83151823557964,37.40019761214057],[55.83264967594742,37.400398790382326],[55.83322180909622,37.39659544313046],[55.83402792148566,37.39667059524539],[55.83638877400216,37.39682089947515],[55.83861656112751,37.39643489154053],[55.84072348043264,37.3955338994751],[55.84502158126453,37.392680272491454],[55.84659117913199,37.39241188227847],[55.84816071336481,37.392529730163616],[55.85288092980303,37.39486835714723],[55.859893456073635,37.39873052645878],[55.86441833633205,37.40272161111449],[55.867579567544375,37.40697072750854],[55.868369880337,37.410007082016016],[55.86920843741314,37.4120992989502],[55.87055369615854,37.412668021163924],[55.87170587948249,37.41482461111453],[55.873183961039565,37.41862266137694],[55.874879126654704,37.42413732540892],[55.875614937236705,37.4312182698669],[55.8762723478417,37.43111093783558],[55.87706546369396,37.43332105622856],[55.87790681284802,37.43385747619623],[55.88027084462084,37.441303050262405],[55.87942070143253,37.44747234260555],[55.88072960917233,37.44716141796871],[55.88121221323979,37.44769797085568],[55.882080694420715,37.45204320500181],[55.882346110794586,37.45673176190186],[55.88252729504517,37.463383999999984],[55.88294937719063,37.46682797486874],[55.88361266759345,37.470014457672086],[55.88546991372396,37.47751410450743],[55.88534929207307,37.47860317658232],[55.882563306475106,37.48165826025772],[55.8815803226785,37.48316434442331],[55.882427612793315,37.483831555817645],[55.88372791409729,37.483182967125686],[55.88495581062434,37.483092277908824],[55.8875561994203,37.4855716508179],[55.887827444039566,37.486440636245746],[55.88897899871799,37.49014203439328],[55.890208937135604,37.493210285705544],[55.891342397444696,37.497512451065035],[55.89174030252967,37.49780744510645],[55.89239745507079,37.49940333499519],[55.89339220941865,37.50018383334346],[55.903869074155224,37.52421672750851],[55.90564076517974,37.52977457672118],[55.90661661218259,37.53503220370484],[55.90714113744566,37.54042858064267],
                                                                    [55.905645048442985,37.54320461007303],[55.906608607018505,37.545686966066306],[55.90788552162358,37.54743976120755],[55.90901557907218,37.55796999999999],[55.91059395704873,37.572711542327866],[55.91073854155573,37.57942799999998],[55.91009969268444,37.58502865872187],[55.90794809960554,37.58739968913264],[55.908713267595054,37.59131567193598],[55.902866854295375,37.612687423278814],[55.90041967242986,37.62348079629517],[55.898141151686396,37.635797880950896],[55.89639275532968,37.649487626983664],[55.89572360207488,37.65619302513125],[55.895295577183965,37.66294133862307],[55.89505457604897,37.66874564418033],[55.89254677027454,37.67375601586915],[55.8947775867987,37.67744661901856],[55.89450045676125,37.688347],[55.89422926332761,37.69480554232789],[55.89322256101114,37.70107096560668],[55.891763491662616,37.705962965606716],[55.889110234998974,37.711885134918205],[55.886577568759876,37.71682005026245],[55.88458159806678,37.7199315476074],[55.882281005794134,37.72234560316464],[55.8809452036196,37.72364385977171],[55.8809722706006,37.725371142837474],[55.88037213862385,37.727870902099546],[55.877941504088696,37.73394330422971],[55.87208120378722,37.745339592590376],[55.86703807949492,37.75525267724611],[55.859821640197474,37.76919976190188],[55.82962968399116,37.827835219574],[55.82575289922351,37.83341438888553],[55.82188784027888,37.83652584655761],[55.81612575504693,37.83809213491821],[55.81460347077685,37.83605359521481],[55.81276696067908,37.83632178569025],[55.811486181656385,37.838623105812026],[55.807329380532785,37.83912198147584],[55.80510270463816,37.839079078033414],[55.79940712529036,37.83965844708251],[55.79131399999368,37.840581150787344],[55.78000432402266,37.84172564285271])

new_coords=[ [i,j] for i,j in zip(np.array(mkad_original)[:, 1], np.array(mkad_original)[:, 0])]

msc_boarders=shape({ "type": "Polygon", "crs":"epsg=4326", "coordinates": [new_coords]})


# In[61]:


gdf_hex = get_h3 (new_coords)


# In[84]:


gdf_scooter_hex = gpd.sjoin(gdf_scooter,gdf_hex, op='within')


# In[81]:


gdf_ws['line_buffer'] = gdf_ws.geometry.buffer(0.01, cap_style=2)


# In[83]:


gdf_ws_scooter = gpd.sjoin(gdf_scooter,gdf_ws.set_geometry('line_buffer'), op='intersects', how='left')


# In[90]:


gdf_ws.osmid1.value_counts().value_counts()


# In[87]:


gdf_ws_scooter


# In[ ]:


дисперсия скорости в рамках одной поездки


# In[ ]:


средняя скорость в зоне, средняя дисперсия скорости в зоне. Слишком медленно плохо, слишком быстро плохо


# In[85]:


gdf_scooter_hex.shape


# In[86]:


gdf_scooter.shape


# In[91]:


gdf_scooter_hex.head()


# In[ ]:


gdf_scooter_hex


# In[93]:


gdf_hex_day_rides = gdf_scooter_hex.groupby(['index_right','gps_date','ride_id']).size().groupby(level=[0,1]).size()


# In[97]:


gdf_hex_day_rides = gdf_hex_day_rides.reset_index()


# In[99]:


plt.hist(gdf_hex_day_rides.groupby('index_right').mean());


# In[102]:


gdf_sim_crash_hex = gpd.sjoin ( gdf_sim_crash,gdf_hex, op='within')


# In[129]:


gdf_hex_day_rides_crashes = gdf_hex_day_rides.groupby('index_right').mean().join(gdf_sim_crash_hex.groupby('index_right').size().to_frame(), on='index_right', how='left', lsuffix='_x',rsuffix='_y').fillna(0)


# In[133]:


gdf_hex_day_rides_crashes.reset_index(inplace=True)


# In[134]:


gdf_hex_day_rides_crashes.rename(columns = {'0_x':'avg_day_rides','0_y':'sum_crashes_2021','index_right':'hex_gid'}, inplace=True)


# In[111]:


days =365+365//2 The #547


# In[136]:


gdf_hex_day_rides_crashes['probability_crash']=100*(gdf_hex_day_rides_crashes['sum_crashes_2021']/(gdf_hex_day_rides_crashes['avg_day_rides']*547))


# In[142]:


plt.hist(gdf_hex_day_rides_crashes['probability_crash'], bins=10);


# In[148]:


pd.qcut(gdf_hex_day_rides_crashes['probability_crash'],q=10,duplicates='drop' )


# In[152]:


np.percentile(gdf_hex_day_rides_crashes['avg_day_rides'], q=90)


# In[138]:


gdf_hex_day_rides_crashes = get_gdf(gdf_hex_day_rides_crashes.merge(gdf_hex.reset_index(), left_on='hex_gid', right_on='index'),
                                    'geometry',
                                    4326)


# In[139]:


gdf_hex_day_rides_crashes.hex_gid.value_counts()


# In[ ]:


gdf_hex_day_rides_crashes


# In[140]:


gdf_hex_day_rides_crashes.to_file('gdf_hex_day_rides_crashes.geojson', driver='GeoJSON')


# In[153]:


# велодорожки


# In[154]:


gdf_velolines = gpd.read_file('bike_lanes.geojson')


# In[186]:


gdf_hex_sample = gdf_hex_day_rides_crashes.merge( gdf_scooter_hex.groupby(['index_right'])['wheel'].mean().reset_index(),
                               left_on='hex_gid',
                               right_on='index_right')


# In[166]:


gdf_hex_sample 


# In[164]:


gdf_scooter_hex.groupby(['index_right','ride_id'])['wheel'].std().groupby(level=0).mean()


# In[187]:


gdf_hex_sample  = gdf_hex_sample.merge( gdf_scooter_hex.groupby(['index_right','ride_id'])['wheel'].std().groupby(level=0).mean().reset_index(),
                               left_on='hex_gid',
                               right_on='index_right',
                                      how='left')


# In[178]:


plt.scatter(gdf_hex_sample['wheel_mean'], gdf_hex_sample['sum_crashes_2021'])


# In[180]:


gdf_hex_sample.groupby(gdf_hex_sample['wheel_mean'].apply(lambda x: x//5*5) )['sum_crashes_2021'].mean().plot(kind='bar')


# In[182]:


gdf_hex_sample.groupby(gdf_hex_sample['wheel_std'].apply(lambda x: x//3*3) )['sum_crashes_2021'].mean().plot(kind='bar')


# In[190]:


gdf_hex_sample_velo = gpd.sjoin(gdf_hex_sample,gdf_velolines, op='intersects', how='left').fillna(0)


# In[191]:


gdf_hex_sample_velo.rename(columns ={'wheel_x':'wheel_mean','wheel_y':'wheel_std'}, inplace=True)


# In[193]:


gdf_hex_sample_velo.groupby(gdf_hex_sample_velo['bikeline_width'])['avg_day_rides'].mean().plot(kind='bar')


# In[194]:


gdf_hex_sample_velo.groupby(gdf_hex_sample_velo['bikeline_width'])['sum_crashes_2021'].mean().plot(kind='bar')


# In[200]:


gdf_hex_sample_velo.drop(['index_right'], axis=1, inplace=True)


# In[201]:


gdf_hex_sample_velo.columns


# In[184]:


gdf_hex_sample['wheel_std'].apply(lambda x: x//3*3).max()


# In[ ]:


plt.scatter(gdf_hex_sample['sum_crashes_2021'],gdf_hex_sample['wheel_std'])


# In[203]:


gdf_hex_sample_velo_osm = gpd.sjoin ( gdf_hex_sample_velo, gdf_ws, op='intersects')


# In[205]:


gdf_hex_sample_velo_ws = gdf_hex_sample_velo.merge(gdf_hex_sample_velo_osm.groupby('hex_gid')['mean'].mean().reset_index(),
                                                   on ='hex_gid')


# In[215]:


df_road_type = gdf_hex_sample_velo_osm.groupby(['hex_gid','highway']).size().to_frame()


# In[225]:


gdf_hex_sample_velo_osm['highway'].value_counts()


# In[219]:


df_road_type['max_rows'] = df_road_type.groupby('hex_gid').transform('max')


# In[221]:


df_road_type_pop = df_road_type[df_road_type['max_rows']==df_road_type[0]].reset_index()[['hex_gid','highway']]


# In[228]:


df_road_type_pop = pd.concat([df_road_type_pop,pd.get_dummies(df_road_type_pop['highway'])],axis=1)


# In[230]:


gdf_hex_sample_velo_ws = gdf_hex_sample_velo_ws.merge(df_road_type_pop ,on='hex_gid')


# In[231]:


df_infra


# In[233]:


gdf_hex_infra = gpd.sjoin(df_infra,gdf_hex, op='intersects', how='inner')


# In[235]:


gdf_hex_infra


# In[238]:


gdf_hex_infra = pd.pivot_table(gdf_hex_infra, index='index_right', columns='Caption', values='global_id', aggfunc='count').reset_index().fillna(0)


# In[ ]:





# In[241]:


gdf_sample_fin = gdf_hex_sample_velo_ws.merge(gdf_hex_infra , left_on='hex_gid', right_on='index_right')


# In[242]:


gdf_sample_fin.to_csv('gdf_sample_fin.csv')


# In[245]:


gdf_sample_fin.groupby('sum_crashes_2021')['mean'].mean().plot(kind='bar')


# In[246]:


gdf_sample_fin.groupby(pd.qcut(gdf_sample_fin['mean'], q=5))['sum_crashes_2021','avg_day_rides','probability_crash'].mean()


# In[252]:


gdf_sample_fin.groupby(pd.qcut(gdf_sample_fin['mean'], q=5))['sum_crashes_2021'].mean().plot(kind='bar')
plt.xlabel('Уровень пешеходности')
plt.ylabel('Среднее число аварий за 3 года')


# In[253]:


from sklearn.ensemble import RandomForestRegressor 


# In[365]:



def get_best_rf_model(X_train_tr,y_train_tr, cv):
    
    max_feature= min(15,X_train_tr.shape[1] )

    # Create the parameter grid based on the results of random search 
    param_grid = {
        'bootstrap': [True],
        'max_depth': np.arange(2,6,1),
        'max_features': [max_feature] ,
        'min_samples_leaf': [5],
        'n_estimators': np.arange(2,10,1)
    }
    # Create a based model
    rf = RandomForestRegressor(random_state=42)
    # Instantiate the grid search model
    grid_search = GridSearchCV(estimator = rf, param_grid = param_grid, 
                              cv = cv, n_jobs = -1, verbose = 2, scoring='neg_mean_squared_error' )

    # Fit the grid search to the data
    grid_search.fit(X_train_tr, y_train_tr)
    print(grid_search.best_params_)

    best_grid = grid_search.best_estimator_

    return best_grid


# In[356]:


import shap


from sklearn.metrics import r2_score


# In[310]:


from sklearn.model_selection import GridSearchCV


# In[311]:


gdf_sample_fin.columns


# In[286]:


gdf_sample_fin.rename(columns={'mean':'walk score'}, inplace=True)


# In[366]:


X_train_tr=gdf_sample_fin[['avg_day_rides','wheel_mean', 'wheel_std', 'bikeline_width',
       'walk score', "['residential'",  'primary',
       'primary_link', 'residential', 'secondary', 'secondary_link',
       'tertiary', 'tertiary_link',  'coffee',
        'grocery',  'metro', 'park', 'ped_area', 'rail',
         'shopping', 'sport', 'university']]
y_train_tr=gdf_sample_fin['probability_crash']
cv=3


# In[367]:


X_train_tr.isnull().mean()


# In[368]:


best_grid = get_best_rf_model(X_train_tr,y_train_tr, cv)


# In[369]:


print(r2_score(y_train_tr,best_grid.predict(X_train_tr)))

ft_imp = pd.DataFrame((zip(X_train_tr.columns,best_grid.feature_importances_)))           .sort_values(1,ascending=True)           .set_index(0)

ft_imp.plot(kind='barh', figsize=(10,8))


# In[384]:


features_core =ft_imp.iloc[-8:].index


# In[385]:


X_train_tr=gdf_sample_fin[features_core]
y_train_tr=gdf_sample_fin['probability_crash']
cv=3


# In[386]:


best_grid = get_best_rf_model(X_train_tr,y_train_tr, cv)


# In[387]:


print(r2_score(y_train_tr,best_grid.predict(X_train_tr)))

ft_imp = pd.DataFrame((zip(X_train_tr.columns,best_grid.feature_importances_)))            .sort_values(1,ascending=True)            .set_index(0)

ft_imp.plot(kind='barh', figsize=(10,8))


# In[388]:



shap_test = shap.TreeExplainer(best_grid).shap_values(X_train_tr[features_core])
shap.summary_plot(shap_test, X_train_tr[features_core],
                      max_display=25, auto_size_plot=True)


# In[ ]:




