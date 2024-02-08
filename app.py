import streamlit as st
import geopandas as gpd
import pandas as pd
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import numpy as np
#import packages
import requests

import warnings
warnings.filterwarnings("ignore")
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import geopandas as gpd
import plotly.express as px
import folium
import json
from datetime import datetime
import plotly.graph_objects as go
from shapely import geometry
from streamlit_folium import st_folium,folium_static
import leafmap.foliumap as leafmap
import streamlit_nested_layout
from folium_legend import add_categorical_legend


st.set_page_config(page_title="Above From the Ponds",page_icon="✅",layout="wide")

import boto3

client = boto3.client(
    's3',
    aws_access_key_id = 'AKIAT2D6RYA7A5PBPXVH',
    aws_secret_access_key = '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',
    region_name = 'ap-south-1'
)


@st.cache(allow_output_mutation = True,suppress_st_warning=True)
def load_data():

    index_stats_df = pd.read_csv("./Data/save_index_version_mar28_v3.csv")
    # obj_index = client.get_object(Bucket='testdevsagemaker', Key='doc_poc/save_index_version_apr06_v1.csv')
    # index_stats_df = pd.read_csv(obj_index['Body'])
    
    # obj_ponds = client.get_object(Bucket='testdevsagemaker', Key='doc_poc/nellore_ponds_df.csv')
    # nellore_ponds_df = pd.read_csv(obj_ponds['Body'])
    
    nellore_ponds_df = pd.read_csv("./Data/nellore_poc_case1.csv")
    nellore_ponds_df.geometry = gpd.GeoSeries.from_wkt(nellore_ponds_df.geometry)
    nellore_ponds_df = gpd.GeoDataFrame(nellore_ponds_df , geometry = nellore_ponds_df.geometry)
    nellore_ponds_df.crs = {'init' :'epsg:4326'}
    nellore_ponds_df["zone"] = nellore_ponds_df["Unnamed: 0"]
    
    # nellore_ponds_df = nellore_ponds_df.head(500)
    
    index_stats_df["time"]= pd.to_datetime(index_stats_df["time"])
    ndwi_data = index_stats_df[index_stats_df["spectral_index"]=="NDWI"]
    ndwi_data = ndwi_data.reset_index()
    ndwi_data = ndwi_data.set_index("time")
    ndwi_ipt = ndwi_data.groupby("zone")["mean"].resample('5D').mean().interpolate('linear').reset_index()

    ndwi_ipt = ndwi_ipt.set_index("time")
    ndwi_ipt_rm = ndwi_ipt.groupby("zone").rolling(5).mean().reset_index()
    ndwi_ipt_rm["status"] = ndwi_ipt_rm["mean"].apply(lambda x : "fallow" if x <-0.05 else "active")


    
    return index_stats_df , nellore_ponds_df ,ndwi_ipt_rm

index_stats_df,nellore_ponds_df,ndwi_ipt_rm =load_data()

cons_active_stats_df = pd.DataFrame()


def get_farmer_persona(zone):
    list_farm_cols = ['Email Address',
       'Farmer/Fisherman Name', 'Phone No', 'Pond Size (Acre)', 'Leased Area',
       'Lease Price/Acre']
    df_filtered = nellore_ponds_df[nellore_ponds_df["zone"]==zone][list_farm_cols]
    
    df_personal = df_filtered.groupby(['Email Address','Farmer/Fisherman Name', 'Phone No']).mean().reset_index()
    df_personal = df_personal.T
    # df_personal.columns=["Value"]
    if df_personal.shape[1]>0:
        df_personal.columns=["Value"]
        return df_filtered 
    else :
        return pd.DataFrame({"Famer Details":["NotAvailable"]})
    

def get_pond_persona(zone):
    
    list_pond_cols = ['zone','Feed Type (Aquaculture)', 'Seed (Aquaculture)',
                   'Conversion Status', 'Trail Net',
                'Financial Dependancy',
                   'Farmer Expectation', 'Insurance', 'Insurance Provider', 'Antibiotics',
                   'List of Antibiotics', 'Pond Certifications', 'List of Certifications',
                   'Distress Sell', 'Distress Sell Remarks', 'Crop Rotation',
                   'Canals And Water Mixing Channels Visibility', 'High Price Trends',
                   'field_38', 'Disease Severity', 'Pond Condition', 'Count/kg',
                   'Pond Activity', 'Growth']

    df_filtered = nellore_ponds_df[nellore_ponds_df["zone"]==zone][list_pond_cols]
    df_filtered = df_filtered.T
    
    if df_filtered.shape[1]>0:
        df_filtered.columns=["Value"]
        return df_filtered 
    else :
        return pd.DataFrame({"Famer Details":["NotAvailable"]})



for time in ndwi_ipt_rm[(ndwi_ipt_rm.time.dt.month<=12) & (ndwi_ipt_rm.time.dt.year<=2023)].time.unique():
    ndwi_time_fil = ndwi_ipt_rm[ndwi_ipt_rm.time==time]
    ndwi_ipt_active = ndwi_time_fil.loc[ndwi_time_fil.groupby(["zone"])['time'].idxmax()]
    temp_df = ndwi_ipt_active.groupby("status")["zone"].count().reset_index().rename(columns={"zone":"n_ponds"})
    temp_df["time"] = time
    cons_active_stats_df=cons_active_stats_df.append(temp_df)



### Feed and Seed Distribution : 
feed_df = nellore_ponds_df.groupby("Feed Type (Aquaculture)")["zone"].count().rename("%Usage").reset_index().groupby(["Feed Type (Aquaculture)"]).sum().transform(lambda x: x/np.sum(x)*100).reset_index()
feed_plot = px.pie(feed_df,values='%Usage', names="Feed Type (Aquaculture)",hole=.3,title="Feed Brands Distribution")

seed_df = nellore_ponds_df.groupby("Seed (Aquaculture)")["zone"].count().rename("%Usage").reset_index().groupby(["Seed (Aquaculture)"]).sum().transform(lambda x: x/np.sum(x)*100).reset_index()
seed_plot = px.pie(seed_df,values='%Usage', names="Seed (Aquaculture)",hole=.3,title="Seed Brands Distribution")
### View 1 : Distribution of Active and Fallow Ponds

active_ponds_plot = px.bar(cons_active_stats_df, x='time', y='n_ponds',
              color='status',
              height=400,
             title="Distribution of Active Ponds over the time")

### Timeline - Readiness

def get_harvest_readiness(doc):
    if doc in range(40,61):
        return "40-60 DoC"
    elif doc in range(61,90):
        return "61-90 DoC"
    elif doc in range(91,120):
        return "91-120 DoC"
    elif doc > 120 :
        return ">120 DoC"
    elif doc in range(5,41) :
        return "Early Stage Culture"
    else :
        return "Dry"
    
def compress(lst):
    res = []
    i = 0
    ind = 0
    while ind < len(lst):
        letter_count = 0
        while i < len(lst) and lst[i] == lst[ind]:
            letter_count += 1
            i +=1
        res.append((lst[ind], letter_count))
        ind += letter_count
    return res



def get_readiness_df(ndwi_ipt_rm):
    df_pnds_profile = ndwi_ipt_rm.copy()

    df_pnds_profile["status"] = df_pnds_profile["mean"].apply(lambda x : "fallow" if x <-0.07 else "active")
    df_pnds_profile["event"] = df_pnds_profile["status"].apply(lambda x : False if x=="active" else True)

    # data_ponds = data_ponds.set_index("time")
    df_pnds_profile['time_of_last_event'] = df_pnds_profile['time'].where(df_pnds_profile['event']).ffill()

    df_pnds_profile["end_date"] = df_pnds_profile.index

    df_pnds_profile["status"] = df_pnds_profile["mean"].apply(lambda x : "fallow" if x <-0.1 else "active")

    df_pnds_profile["start_date"] = df_pnds_profile["end_date"].shift(1)
    df_pnds_profile = df_pnds_profile.set_index("time")

    df_pnds_profile["proj_DoC"]= (df_pnds_profile.index-df_pnds_profile["time_of_last_event"]).dt.days
    
    df_pnds_profile_10d = df_pnds_profile[df_pnds_profile.index=="2023-04-05"]
    df_pnds_profile_10d["proj_DoC_10d"]= df_pnds_profile_10d["proj_DoC"].apply(lambda x : x+10 if x>0 else 0)
    df_pnds_profile_10d["proj_DoC_20d"]= df_pnds_profile_10d["proj_DoC"].apply(lambda x : x+20 if x>0 else 0)
    df_pnds_profile_10d["harvest_readiness_10d"] = df_pnds_profile_10d["proj_DoC_10d"].apply(lambda x : get_harvest_readiness(x))
    df_pnds_profile_10d["harvest_readiness_20d"] = df_pnds_profile_10d["proj_DoC_20d"].apply(lambda x : get_harvest_readiness(x))
    
    df_grouped_hr_10d = df_pnds_profile_10d.groupby("harvest_readiness_10d")["zone"].count().rename("n_Ponds").reset_index()
    df_grouped_hr_20d = df_pnds_profile_10d.groupby("harvest_readiness_20d")["zone"].count().rename("n_Ponds").reset_index()

    df_grouped_hr_10d = df_grouped_hr_10d[df_grouped_hr_10d["harvest_readiness_10d"].isin(["40-60 DoC","61-90 DoC","91-120 DoC"])]
    df_grouped_hr_20d = df_grouped_hr_20d[df_grouped_hr_20d["harvest_readiness_20d"].isin(["40-60 DoC","61-90 DoC","91-120 DoC"])]

    return df_pnds_profile_10d,df_grouped_hr_10d,df_grouped_hr_20d
    

df_pnds_profile_10d,df_hr_10d,df_hr_20d = get_readiness_df(ndwi_ipt_rm)

hr_10_plot = px.bar(df_hr_10d, x="n_Ponds", y="harvest_readiness_10d", orientation='h',title="Harvest Readiness - 10 days from 31st March")
hr_20_plot = px.bar(df_hr_20d, x="n_Ponds", y="harvest_readiness_20d", orientation='h',title="Harvest Readiness - 20 days from 31st March")



def get_pond_practice_df(list_event):
    
    list_cycles = []
    for idx in range(0,len(list_event)):
        if idx>0:
            event , duration = list_event[idx]
            event_p , duration_p = list_event[idx-1]
            # event_f , duration_f = list_event[idx]


            # print(event,duration)
            # print(event_p , duration_p)
            if (event_p == "F") :
                if (duration_p>15):
                    if duration<=120:
                        list_cycles.append((event_p,duration_p,event,duration,"Dried Before Stocking","Good Active DoC cycle"))
                    else :
                        list_cycles.append((event_p,duration_p,event,duration,"Dried Before Stocking","Possible Crop Rotation"))

                else :
                    if duration<=120:
                        list_cycles.append((event_p,duration_p,event,duration,"Inadequate Drying Before Stocking","Good Active DoC cycle"))
                    else :
                        list_cycles.append((event_p,duration_p,event,duration,"Inadequate Drying Before Stocking","Possible Crop Rotation"))
                        
    pond_practice_df = pd.DataFrame(list_cycles,columns=["Fallow","FallowPeriod","Active","ActivePeriod","Drying Practice","Culture_Practice"])
    pond_practice_df["Cycle"] = pond_practice_df.index+1
    return pond_practice_df


def get_pond_practice_all(ndwi_ipt_rm):
    pond_practice_df_all = pd.DataFrame()

    for pond_no in ndwi_ipt_rm["zone"].unique():
        data_seq = ndwi_ipt_rm[["zone","time","status"]]
        data_seq = data_seq[data_seq["zone"]==pond_no]
        data_seq["status_flag"] = data_seq["status"].apply(lambda x : "F" if x=="fallow" else "A")
        data_seq_1 = data_seq[["time","status_flag"]].set_index("time").resample("1D").last().ffill()
        list_event = compress(data_seq_1["status_flag"])
        pond_practice_df = get_pond_practice_df(list_event)
        pond_practice_df["zone"] = pond_no
        pond_practice_df_all = pond_practice_df_all.append(pond_practice_df)
    pond_practice_df_all = pond_practice_df_all.reset_index()
    return pond_practice_df_all

pond_practice_df_all = get_pond_practice_all(ndwi_ipt_rm)

latest_pond_practive = pond_practice_df_all.groupby("zone").tail(1)

dry_prac_df = latest_pond_practive.groupby("Drying Practice")["Culture_Practice"].count().rename("%Ponds").reset_index().groupby(["Drying Practice"]).sum().transform(lambda x: x/np.sum(x)*100).reset_index()

cul_prac_df = latest_pond_practive.groupby("Culture_Practice")["Culture_Practice"].count().rename("%Ponds").reset_index().groupby(["Culture_Practice"]).sum().transform(lambda x: x/np.sum(x)*100).reset_index()


cul_plot = px.pie(cul_prac_df,values='%Ponds', names="Culture_Practice",hole=.3,title="Culture Practice Distribution")
dry_plot = px.pie(dry_prac_df,values='%Ponds', names="Drying Practice",hole=.3,title="Drying Practice Distribution")


####---- Loading Map ---------------- ###
def load_map(nellore_ponds_df,ndwi_ipt_rm,zone):
    
    df_zoom = gpd.GeoDataFrame(nellore_ponds_df[nellore_ponds_df.zone==zone])
    x,y = df_zoom["geometry"].centroid.values.x[0],df_zoom["geometry"].centroid.values.y[0]

    map_test = folium.Map(location=[y,x],tiles="openstreetmap",   zoom_start=25)
    # plotting the Ponds available in side the study region
    basemaps = {
                    'Esri Satellite': folium.TileLayer(
                        tiles = 'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
                        attr = 'Esri',
                        name = 'Esri Satellite',
                        overlay = True,
                        control = True
                    )
                }

    nellore_ponds_df= gpd.GeoDataFrame(nellore_ponds_df[nellore_ponds_df.geometry!=None])
    # nellore_ponds_df = nellore_ponds_df.set_crs("epsg:4326")
    ndwi_ipt_active_recent = ndwi_ipt_rm.loc[ndwi_ipt_rm.groupby(["zone"])['time'].idxmax()]
    
    active_ponds = ndwi_ipt_active_recent[ndwi_ipt_active_recent["status"]=="active"]["zone"].unique()
    
    active_ponds_df = nellore_ponds_df[nellore_ponds_df.zone.isin(active_ponds)]
    fallow_ponds  = ndwi_ipt_active_recent[ndwi_ipt_active_recent["status"]!="active"]["zone"].unique()
    fallow_ponds_df = nellore_ponds_df[nellore_ponds_df.zone.isin(fallow_ponds)]

    tooltip1=folium.features.GeoJsonTooltip(fields=["zone","Species","Farmer/Fisherman Name","Phone No"])


    folium.GeoJson(data=active_ponds_df[["geometry","Species","zone","Farmer/Fisherman Name","Phone No"]],
                    style_function = lambda x : {'color': "blue",'fillColor': '#00000000',"fill_opacity":1}
                ,name='Active Ponds (Recent)',
                ).add_to(map_test)

    folium.GeoJson(data=fallow_ponds_df[["geometry","Species","zone","Farmer/Fisherman Name","Phone No"]],
                    style_function = lambda x : {'color': "red",'fillColor': '#00000000',"fill_opacity":1}
                ,name='Fallow Ponds (Recent)',
                tooltip=tooltip1).add_to(map_test)

    map_test = add_categorical_legend(map_test,'My title 2',colors = ['#FF','#777'],labels = ['Fallow', 'Active'])
    basemaps['Esri Satellite'].add_to(map_test)
    folium.LayerControl().add_to(map_test)
    map_test = add_categorical_legend(map_test, 'My Ff 2',
                             colors = ['red','#777'],
                           labels = ['Heat 2', 'Cold 2'])

    return map_test



def get_weather_graphs(lat,lon):
    url = "https://api.openweathermap.org/data/3.0/onecall?lat={}&lon={}&exclude=minutely&appid=32ff182db516cfc764a55ad998d76117".format(lat,lon)
    res = requests.get(url)
    data = res.json()
    weather_forecast_7 = pd.DataFrame.from_records(data["daily"])
    weather_forecast_7["temp_day"] = weather_forecast_7.temp.apply(lambda x: x["day"])
    # weather_forecast_7["temp_night"] = weather_forecast_7.temp.apply(lambda x: x["night"])
    weather_forecast_7["temp_max"] = weather_forecast_7.temp.apply(lambda x: x["max"])
    weather_forecast_7["temp_min"] = weather_forecast_7.temp.apply(lambda x: x["min"])
    weather_forecast_7["date"] = weather_forecast_7["dt"].apply(lambda x : datetime.fromtimestamp(x))
    weather_plot=weather_forecast_7[["temp_day","temp_max","temp_min","pop","date"]]
    
    # weather_forecast_7 = weather_forecast_7.set_index("date")
    fig_pop = px.line(weather_plot,weather_plot.date,weather_plot["pop"]*100,title="Forecast - 7days - Precipitation")
    fig_pop.update_yaxes(title="Propability of Precipitaiton(%)")
    
    fig_temp = px.line(weather_plot,weather_plot.date,weather_plot["temp_max"]-273,title="Forecast - 7days - Temperature")
    fig_temp.update_yaxes(title="Temperature in °C")

    return fig_pop, fig_temp

def get_lat_lon(fidnum):
    # lat = nellore_ponds_df[nellore_ponds_df.zone==fidnum]["Latitude"].item()
    # lon = nellore_ponds_df[nellore_ponds_df.zone==fidnum]["Longitude"].item()
    lat =nellore_ponds_df[nellore_ponds_df.zone==fidnum]["geometry"].item().centroid.y
    lon=nellore_ponds_df[nellore_ponds_df.zone==fidnum]["geometry"].item().centroid.x
    return lat , lon

# columns = ["geometry","boundaries","species"]

# folium.GeoJson(data=nellore_ponds_df,
#                style_function=lambda x: {'color':"red"},
#                name='Zorp_Ponds_matchs_with_boundary',
#                tooltip=folium.features.GeoJsonTooltip(
#              fields=["zone"])).add_to(map_test)




st.title("Shrimp360 - Decision Board")

padding_top = 0
st.write('<style>div.block-container{padding-top:1rem;padding-bottom:1rem;}</style>', unsafe_allow_html=True)

st.write('NB : Date Range considered for PoC : 01 Jan 2022 - 04 April 2023 : subject to Sentientl 2 Data availability')

st.info("2023 is going to be an el-nino year. El Niño is a weather phenomenon that can cause changes in water temperature and weather patterns, potentially reducing shrimp yields in India. Shrimp farmers should be aware of these impacts and take appropriate measures to mitigate risks")

# st.write("Welcome to Aqua360 Decision Board")
st.plotly_chart(active_ponds_plot, theme="streamlit", use_container_width=True)

cola, colb = st.columns([2,2])

with cola:
    st.write("What are our farmers using?")
    cola1,cola2 = st.columns([2,2])
    with cola1:
        st.plotly_chart(feed_plot, theme="streamlit", use_container_width=True)

    with cola2:
        st.plotly_chart(seed_plot, theme="streamlit", use_container_width=True)

with colb:
    st.write("Harvest Readiness - 10 to 20 days down the line")

    colb1,colb2 = st.columns([2,2])
    with colb1:
        st.plotly_chart(hr_10_plot, theme="streamlit", use_container_width=True)

    with colb2:
        st.plotly_chart(hr_20_plot, theme="streamlit", use_container_width=True)


st.write("Ponds Following Best Practices: ")
coli, colj= st.columns([2,2])


with coli:
        st.plotly_chart(dry_plot, theme="streamlit", use_container_width=True)

with colj:
        st.plotly_chart(cul_plot, theme="streamlit", use_container_width=True)
 
        

col1, col2 = st.columns([1,4])

col3, col4= st.columns([2,2])


with col1:
    show_pond = option = st.selectbox('Select Certain type of Ponds?',('All', 'active', 'fallow'))

    df_active_options = ndwi_ipt_rm.loc[ndwi_ipt_rm.groupby(["zone"])['time'].idxmax()]

    if show_pond == "All":
        ponds_opts = df_active_options["zone"].unique()
        ponds_opts = [x for x in ponds_opts if x!=0]
        # selected_pond  = st.selectbox('Select the Pond to View',ponds_opts)

         
    if show_pond == "active":
#         active_pond_opts = df_pnds_profile_10d["harvest_readiness_10d"].unique()
#         active_pond_opts = [x for x in active_pond_opts if x!="Dry"]
#         # selected_status =  st.selectbox('Select the Pond to View',active_pond_opts)

#         ponds_opts = df_pnds_profile_10d[df_pnds_profile_10d["harvest_readiness_10d"]==selected_status]["zone"].unique()
#         ponds_opts = [x for x in ponds_opts if x!=0]
        
#         # selected_pond  = st.selectbox('Select the Pond to View',ponds_opts)
        ponds_opts = df_active_options[df_active_options["status"]==show_pond]["zone"].unique()
        ponds_opts = [x for x in ponds_opts if x!=0]
        ponds_opts = np.unique(ponds_opts)


    if show_pond == "fallow":
        ponds_opts = df_active_options[df_active_options["status"]==show_pond]["zone"].unique()
        ponds_opts = [x for x in ponds_opts if x!=0]
        ponds_opts = np.unique(ponds_opts)

        # selected_pond  = st.selectbox('Select the Pond to View',ponds_opts)
    selected_pond  = st.selectbox('Select the Pond to View',ponds_opts)




    
with col2:
    
    st.write("Spatial View of Ponds")
    st.info('Red - Fallow Ponds as of 31st March', icon="🔴")
    st.info('Blue - Active Ponds as of 31st March', icon="🔵")

    if selected_pond:
        map_test = load_map(nellore_ponds_df,ndwi_ipt_rm,selected_pond)
        clicked_location=st_folium(map_test, height=500,width=1000)
    else:
        map_test = load_map(nellore_ponds_df,ndwi_ipt_rm,692)
        clicked_location=st_folium(map_test, height=500,width=1000)

        # clicked_location = st_folium_get_lat_lon(map_test)

        # print(clicked_location)
        # st.markdown(map_test._repr_html_(escape=False),unsafe_allow_html=True)
# print(output["last_object_clicked"])
    if clicked_location["last_object_clicked"] != None:
        lat = clicked_location["last_object_clicked"]['lat']
        lan = clicked_location["last_object_clicked"]['lng']
        
        fidnum = nellore_ponds_df[nellore_ponds_df.contains(geometry.Point(lan,lat))]["zone"].to_list()[0]
        lat,lon = get_lat_lon(fidnum)

        
        data  = ndwi_ipt_rm
        data_ponds = data[data["zone"]==fidnum]
        data_ponds["status"] = data_ponds["mean"].apply(lambda x : "fallow" if x <-0.07 else "active")
        data_ponds["event"] = data_ponds["status"].apply(lambda x : False if x=="active" else True)

        # data_ponds = data_ponds.set_index("time")
        data_ponds['time_of_last_event'] = data_ponds['time'].where(data_ponds['event']).ffill()
        data_ponds= data_ponds.set_index("time")
        data_ponds["end_date"] = data_ponds.index

        # data_ponds["status"] = data_ponds["mean"].apply(lambda x : "fallow" if x <-0.1 else "active")

        data_ponds["start_date"] = data_ponds["end_date"].shift(1)
        data_ponds["proj_DoC"]= (data_ponds.index-data_ponds["time_of_last_event"]).dt.days



    else :

        data  = ndwi_ipt_rm
        fidnum = selected_pond
        lat,lon = get_lat_lon(fidnum)

        data_ponds = data[data["zone"]==fidnum]
        data_ponds["status"] = data_ponds["mean"].apply(lambda x : "fallow" if x <-0.07 else "active")
        data_ponds["event"] = data_ponds["status"].apply(lambda x : False if x=="active" else True)

        # data_ponds = data_ponds.set_index("time")
        data_ponds['time_of_last_event'] = data_ponds['time'].where(data_ponds['event']).ffill()
        data_ponds= data_ponds.set_index("time")
        data_ponds["end_date"] = data_ponds.index

        # data_ponds["status"] = data_ponds["mean"].apply(lambda x : "fallow" if x <-0.1 else "active")

        data_ponds["start_date"] = data_ponds["end_date"].shift(1)
        data_ponds["proj_DoC"]= (data_ponds.index-data_ponds["time_of_last_event"]).dt.days


with col3:
    st.write("Pond Lifecycle of Pond : "+str(fidnum))
    doc_timeline = px.timeline(data_ponds, x_start="start_date", x_end="end_date", y="status",color="status")
    st.plotly_chart(doc_timeline, theme="streamlit", use_container_width=True)
    doc_projected = px.line(data_ponds, x=data_ponds.index, y="proj_DoC",)
    st.write("Projected DoC : "+str(fidnum))
    st.plotly_chart(doc_projected, theme="streamlit", use_container_width=True)

with col4:
    ndwi_plot = px.scatter(data_ponds, x=data_ponds.index, y="mean", color='zone',trendline="rolling", trendline_options=dict(window=5))
    ndwi_plot.update_layout(shapes=[
    dict(
    type= 'line',
    yref= 'y', y0= -0.05, y1= -0.05,   # adding a horizontal line at Y = 1
    xref= 'paper', x0= 0, x1= 1) 
    ])
    st.write("NDWI over time : "+str(fidnum))
    st.plotly_chart(ndwi_plot, theme="streamlit", use_container_width=True)

    farmer_persona = get_farmer_persona(fidnum)

    st.write ("Famer Details")
    st.table(farmer_persona)




colt,colp = st.columns([2,2])

fig_temp , fig_pop = get_weather_graphs(lat,lon)

with colt:
    st.plotly_chart(fig_temp, theme="streamlit", use_container_width=True)

with colp:
    st.plotly_chart(fig_pop, theme="streamlit", use_container_width=True)




