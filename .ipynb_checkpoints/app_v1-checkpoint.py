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


@st.cache(allow_output_mutation = True,suppress_st_warning=True)
def load_data():

    index_stats_df = pd.read_csv("../CF_pond_app/Data/ndwi_stats_2021_24.csv")
    index_stats_df = index_stats_df[index_stats_df['zone']!=22]
    # cf_ponds_df = gpd.read_file("../CF_pond_app/Data/combined.geojson")[:22]
    
    cf_ponds_df = gpd.read_file("../CF_pond_app/Data/combined_v1.geojson").to_crs(4326)[:22]
    cf_ponds_df = cf_ponds_df.rename(columns={'pond name': 'Pond name'})
    
    index_stats_df["time"]= pd.to_datetime(index_stats_df["time"])
    ndwi_data = index_stats_df[index_stats_df["spectral_index"]=="NDWI"]
    ndwi_data = ndwi_data.reset_index()
    ndwi_data = ndwi_data.set_index("time")
    ndwi_ipt = ndwi_data.groupby("zone")["mean"].resample('5D').mean().interpolate('linear').reset_index()

    ndwi_ipt = ndwi_ipt.set_index("time")
    ndwi_ipt_rm = ndwi_ipt.groupby("zone").rolling(5).mean().reset_index()

    ndwi_ipt_rm = ndwi_ipt_rm.set_index("time")
    ndwi_ipt_rm["end_date"] = ndwi_ipt_rm.index
    ndwi_ipt_rm["start_date"] = ndwi_ipt_rm["end_date"].shift(1)

    # ndwi_ipt_rm = ndwi_ipt_rm.set_index("time")

    ndwi_ipt_rm["status"] = ndwi_ipt_rm["mean"].apply(lambda x : "fallow" if x <-0.05 else "active")
    ndwi_ipt_rm["event"] = ndwi_ipt_rm["status"].apply(lambda x : False if x=="active" else True)
    ndwi_ipt_rm['time'] = pd.to_datetime(ndwi_ipt_rm.index)

    # data_ponds = data_ponds.set_index("time")
    ndwi_ipt_rm['time_of_last_event'] = ndwi_ipt_rm['time'].where(ndwi_ipt_rm['event']).ffill()

    cons_active_stats_df = pd.DataFrame()

    for time in ndwi_ipt_rm[(ndwi_ipt_rm.time.dt.year>=2022)].time.unique():
        ndwi_time_fil = ndwi_ipt_rm[ndwi_ipt_rm.time==time]
        temp_df = ndwi_time_fil.groupby("status")["zone"].count().reset_index().rename(columns={"zone":"n_ponds"})
        temp_df["time"] = time
        cons_active_stats_df=cons_active_stats_df.append(temp_df)


    
    return cons_active_stats_df , cf_ponds_df ,ndwi_ipt_rm

cons_active_stats_df,cf_ponds_df,ndwi_ipt_rm =load_data()

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
    df_pnds_profile = ndwi_ipt_rm[(ndwi_ipt_rm.time.dt.year>=2022)].copy()

    df_pnds_profile["proj_DoC"]= (df_pnds_profile.index-df_pnds_profile["time_of_last_event"]).dt.days
    
    df_pnds_profile_10d = df_pnds_profile[df_pnds_profile.index=="2024-02-19"]
    df_pnds_profile_10d["proj_DoC_10d"]= df_pnds_profile_10d["proj_DoC"].apply(lambda x : x+10 if x>0 else 0)
    df_pnds_profile_10d["proj_DoC_20d"]= df_pnds_profile_10d["proj_DoC"].apply(lambda x : x+20 if x>0 else 0)
    df_pnds_profile_10d["harvest_readiness_10d"] = df_pnds_profile_10d["proj_DoC_10d"].apply(lambda x : get_harvest_readiness(x))
    df_pnds_profile_10d["harvest_readiness_20d"] = df_pnds_profile_10d["proj_DoC_20d"].apply(lambda x : get_harvest_readiness(x))
    
    df_grouped_hr_10d = df_pnds_profile_10d.groupby("harvest_readiness_10d")["zone"].count().rename("n_Ponds").reset_index()
    df_grouped_hr_20d = df_pnds_profile_10d.groupby("harvest_readiness_20d")["zone"].count().rename("n_Ponds").reset_index()

    df_grouped_hr_10d = df_grouped_hr_10d[df_grouped_hr_10d["harvest_readiness_10d"].isin(["40-60 DoC","61-90 DoC","91-120 DoC"])]
    df_grouped_hr_20d = df_grouped_hr_20d[df_grouped_hr_20d["harvest_readiness_20d"].isin(["40-60 DoC","61-90 DoC","91-120 DoC"])]

    return df_pnds_profile,df_pnds_profile_10d,df_grouped_hr_10d,df_grouped_hr_20d
    

df_pnds_profile,df_pnds_profile_10d,df_hr_10d,df_hr_20d = get_readiness_df(ndwi_ipt_rm)

hr_10_plot = px.bar(df_hr_10d, x="n_Ponds", y="harvest_readiness_10d", orientation='h',title="Harvest Readiness - 10 days from 19th February")
hr_20_plot = px.bar(df_hr_20d, x="n_Ponds", y="harvest_readiness_20d", orientation='h',title="Harvest Readiness - 20 days from 19th February")



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
def load_map(cf_ponds_df,ndwi_ipt_rm,zone):
    
    df_zoom = gpd.GeoDataFrame(cf_ponds_df[cf_ponds_df['Pond name']==zone])
    x,y = df_zoom["geometry"].centroid.values.x[0],df_zoom["geometry"].centroid.values.y[0]

    map_test = folium.Map(location=[y,x],tiles="openstreetmap",   zoom_start=25)
    # plotting the Ponds available in side the study region
    basemaps = {
                    'Google Satellite': folium.TileLayer(
                        tiles = 'https://mt1.google.com/vt/lyrs=y&x={x}&y={y}&z={z}',
                        attr = 'Google',
                        name = 'Google Satellite',
                        overlay = True,
                        control = True
                    )
                }

    cf_ponds_df= gpd.GeoDataFrame(cf_ponds_df[cf_ponds_df.geometry!=None])
    ndwi_ipt_active_recent = ndwi_ipt_rm.loc[ndwi_ipt_rm.groupby(["zone"])['time'].idxmax()]
    
    active_ponds = ndwi_ipt_active_recent[ndwi_ipt_active_recent["status"]=="active"]["zone"].unique()
    
    active_ponds_df = cf_ponds_df[cf_ponds_df.fid.isin(active_ponds)]
    fallow_ponds  = ndwi_ipt_active_recent[ndwi_ipt_active_recent["status"]!="active"]["zone"].unique()
    fallow_ponds_df = cf_ponds_df[cf_ponds_df.fid.isin(fallow_ponds)]
    
    tooltip=folium.features.GeoJsonTooltip(fields=['Pond name'])
    tooltip1=folium.features.GeoJsonTooltip(fields=['Pond name'])
    folium.GeoJson(data=active_ponds_df[["geometry",'Pond name']],
                    style_function = lambda x : {'color': "blue",'fillColor': '#00000000',"fill_opacity":1}
                ,name='Active Ponds (Recent)',
                   tooltip=tooltip
                  ).add_to(map_test)

    folium.GeoJson(data=fallow_ponds_df[["geometry",'Pond name']],
                    style_function = lambda x : {'color': "red",'fillColor': '#00000000',"fill_opacity":1}
                ,name='Fallow Ponds (Recent)',
                   tooltip=tooltip1
                  ).add_to(map_test)

    map_test = add_categorical_legend(map_test,'My title 2',colors = ['#FF','#777'],labels = ['Fallow', 'Active'])
    basemaps['Google Satellite'].add_to(map_test)
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
    lat =cf_ponds_df[cf_ponds_df.fid==fidnum]["geometry"].item().centroid.y
    lon=cf_ponds_df[cf_ponds_df.fid==fidnum]["geometry"].item().centroid.x
    return lat , lon

def get_pn(fidnum):
    name =cf_ponds_df[cf_ponds_df.fid==fidnum]["Pond name"].values[0]
    return name


st.title("Shrimp360 - Decision Board")

padding_top = 0
st.write('<style>div.block-container{padding-top:1rem;padding-bottom:1rem;}</style>', unsafe_allow_html=True)

st.write('NB : Date Range considered : 01 Jan 2022 - 19 Feb 2024 : subject to Sentientl 2 Data availability')

st.info("2023 is going to be an el-nino year. El Niño is a weather phenomenon that can cause changes in water temperature and weather patterns, potentially reducing shrimp yields in India. Shrimp farmers should be aware of these impacts and take appropriate measures to mitigate risks")

# st.write("Welcome to Aqua360 Decision Board")
st.plotly_chart(active_ponds_plot, theme="streamlit", use_container_width=True)

cola, colb = st.columns([2,2])

with cola:
    st.write("Ponds Following Best Practices: ")
    cola1,cola2 = st.columns([2,2])
    with cola1:
        st.plotly_chart(dry_plot, theme="streamlit", use_container_width=True)

    with cola2:
        st.plotly_chart(cul_plot, theme="streamlit", use_container_width=True)

with colb:
          
    st.write("Harvest Readiness - 10 to 20 days down the line")

    colb1,colb2 = st.columns([2,2])
    with colb1:
        st.plotly_chart(hr_10_plot, theme="streamlit", use_container_width=True)

    with colb2:
        st.plotly_chart(hr_20_plot, theme="streamlit", use_container_width=True)
        
col1, col2 = st.columns([1,4])

col3, col4= st.columns([2,2])


with col1:
    show_pond = option = st.selectbox('Select Certain type of Ponds?',('All', 'active', 'fallow'))
    df_active_options = ndwi_ipt_rm.loc[ndwi_ipt_rm.groupby(["zone"])['time'].idxmax()]
    df_active_options = df_active_options.merge(cf_ponds_df,left_on='zone',right_on='fid',)

    if show_pond == "All":
        ponds_opts = df_active_options["Pond name"].unique()
        ponds_opts = [x for x in ponds_opts if x!=0]
         
    if show_pond == "active":
        ponds_opts = df_active_options[df_active_options["status"]==show_pond]["Pond name"].unique()
        ponds_opts = [x for x in ponds_opts if x!=0]
        ponds_opts = np.unique(ponds_opts)

    if show_pond == "fallow":
        ponds_opts = df_active_options[df_active_options["status"]==show_pond]["Pond name"].unique()
        ponds_opts = [x for x in ponds_opts if x!=0]
        ponds_opts = np.unique(ponds_opts)

        # selected_pond  = st.selectbox('Select the Pond to View',ponds_opts)
    selected_pond  = st.selectbox('Select the Pond to View',ponds_opts)
    
with col2:
    
    st.write("Spatial View of Ponds")
    st.info('Red - Fallow Ponds as of 19th February', icon="🔴")
    st.info('Blue - Active Ponds as of 19th February', icon="🔵")

    if selected_pond:
        map_test = load_map(cf_ponds_df,ndwi_ipt_rm,selected_pond)
        clicked_location=st_folium(map_test, height=800,width=1500)
    else:
        map_test = load_map(cf_ponds_df,ndwi_ipt_rm,'A1')
        clicked_location=st_folium(map_test, height=800,width=1500)
        
        
    if clicked_location["last_object_clicked"] != None:
        lat = clicked_location["last_object_clicked"]['lat']
        lan = clicked_location["last_object_clicked"]['lng']
        
        fidnum = cf_ponds_df[cf_ponds_df.contains(geometry.Point(lan,lat))]["fid"].to_list()[0]
        lat,lon = get_lat_lon(fidnum)
        p_name = get_pn(fidnum)

        
        data  = df_pnds_profile
        data_ponds = data[data["zone"]==fidnum]
        data_ponds["proj_DoC"]= (data_ponds.index-data_ponds["time_of_last_event"]).dt.days


    else :

        data  = df_pnds_profile
        fidnum = 0
        lat,lon = get_lat_lon(fidnum)
        p_name = get_pn(fidnum) 

        data_ponds = data[data["zone"]==fidnum]
        data_ponds["proj_DoC"]= (data_ponds.index-data_ponds["time_of_last_event"]).dt.days


with col3:
    st.write("Pond Lifecycle of Pond : "+str(p_name))
    doc_timeline = px.timeline(data_ponds, x_start="start_date", x_end="end_date", y="status",color="status")
    st.plotly_chart(doc_timeline, theme="streamlit", use_container_width=True)

with col4:
    doc_projected = px.line(data_ponds, x=data_ponds.index, y="proj_DoC",)
    st.write("Projected DoC : "+str(p_name))
    st.plotly_chart(doc_projected, theme="streamlit", use_container_width=True)
    
ndwi_plot = px.scatter(data_ponds, x=data_ponds.index, y="mean", color='zone',trendline="rolling", trendline_options=dict(window=5))
ndwi_plot.update_layout(shapes=[
dict(
type= 'line',
yref= 'y', y0= -0.05, y1= -0.05,   # adding a horizontal line at Y = 1
xref= 'paper', x0= 0, x1= 1) 
])
st.write("NDWI over time : "+str(p_name))
st.plotly_chart(ndwi_plot, theme="streamlit", use_container_width=True)


colt,colp = st.columns([2,2])

fig_temp , fig_pop = get_weather_graphs(lat,lon)

with colt:
    st.plotly_chart(fig_temp, theme="streamlit", use_container_width=True)

with colp:
    st.plotly_chart(fig_pop, theme="streamlit", use_container_width=True)