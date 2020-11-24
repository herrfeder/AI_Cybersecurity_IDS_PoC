import plotly.graph_objects as go
from plotly.subplots import make_subplots
from statsmodels.tsa.seasonal import seasonal_decompose
from pandas.tseries.offsets import DateOffset
import pandas as pd
import numpy as np
import dash_table
from ipdb import set_trace

import numpy as np           
from scipy.io import netcdf  
#from mpl_toolkits.basemap import Basemap
from numpy import pi, sin, cos




def apply_layout(fig, title="", height=1250):
    '''
    Applying web app layout for plots.
    
    INPUT:
        title - (str) Figure Title
        height - (int) Figure Height 
    OUTPUT:
        fig - (plotly Figure) plot object that needs to be passed to dash figure attribute
    '''
    
    
    fig.update_layout(height=height, title_text=title)
    layout = fig["layout"]
    layout["paper_bgcolor"] = "#002b36"   
    layout["plot_bgcolor"] = "#1f2630"
    layout["font"]["color"] = "#2cfec1"
    layout["title"]["font"]["color"] = "#2cfec1"
    layout["xaxis"]["tickfont"]["color"] = "#2cfec1"
    layout["yaxis"]["tickfont"]["color"] = "#2cfec1"
    layout["xaxis"]["gridcolor"] = "#5b5b5b"
    layout["yaxis"]["gridcolor"] = "#5b5b5b"
    layout["margin"]["t"] = 75
    layout["margin"]["r"] = 0
    layout["margin"]["b"] = 0
    layout["margin"]["l"] = 0

    return fig

def plot_ten_most_ip(data_dict, title="", dash=False):

    ips=list(data_dict.keys())[::-1]
    number=list(data_dict.values())[::-1]

    fig = go.Figure([go.Bar(x=number, y=ips, orientation='h', marker_color="#ff0000", width=0.5)])

    if dash:
        return apply_layout(fig, height=400)
    else:
        fig.show()


def temp_style():

    df['Rating'] = df['Humidity'].apply(lambda x:
    '⭐⭐⭐' if x > 30 else (
    '⭐⭐' if x > 20 else (
    '⭐' if x > 10 else ''   
    )))
    df['Growth'] = df['Temperature'].apply(lambda x: '↗️' if x > 0 else '↘️')
    df['Status'] = df['Temperature'].apply(lambda x: '🔥' if x > 0 else '🚒')
    app.layout = dash_table.DataTable(
    data=df.to_dict('records'),
    columns=[
        {"name": i, "id": i} for i in df.columns
    ],)


def get_data_table_2(df, fig="", title=""):

    dt = dash_table.DataTable(
        id='datatable-row-ids',
        columns=[
            {'name': i, 'id': i, 'deletable': True} for i in df.columns
            # omit the id column
            if i != 'id'
            ],
        data=df.to_dict('records'),
        editable=True,
        filter_action="native",
        sort_action="native",
        sort_mode='multi',
        row_selectable='multi',
        row_deletable=True,
        selected_rows=[],
        page_action='native',
        page_current= 0,
        page_size= 100,
        style_as_list_view=True,
        style_data_conditional=[
            {
                'if': {'row_index': 'odd'},
                'backgroundColor': 'rgb(50, 50, 50)'
            },
            {
                'if': {'row_index': 'even'},
                'backgroundColor': 'rgb(50, 60, 50)'
            },

        ],
        style_header={
            'backgroundColor': 'rgb(140, 0, 0)',
            'fontWeight': 'bold'
            },
        style_cell={ 'border': '10px solid #222' },
        style_table= {'overflowY': 'auto'}
    )

    
   
    return dt

def get_data_table(df, fig="", title="", dash=False):
  
    df = pd.read_csv('https://raw.githubusercontent.com/plotly/datasets/master/2014_usa_states.csv')

    fig = go.Figure(data=[go.Table(row_deletable=True,
        header=dict(values=list(df.columns),
                    line_color='#303030',
                    line={'width':10},
                    height=30,  
                    fill_color='grey',
                    align='center'),
        cells=dict(values=[df.Rank, df.State, df.Postal, df.Population],
                line_color='#303030',
                line={'width':10},
                height=30,  
                fill_color='grey',
                align='center'))
    ])

    if dash:
        return apply_layout(fig, title)
    else:
        fig.show()


# Make shortcut to Basemap object, 
# not specifying projection type for this example
#m = Basemap() 


# Functions converting coastline/country polygons to lon/lat traces
def polygons_to_traces(poly_paths, N_poly):
    ''' 
    pos arg 1. (poly_paths): paths to polygons
    pos arg 2. (N_poly): number of polygon to convert
    '''
    # init. plotting list
    lons=[]
    lats=[]

    for i_poly in range(N_poly):
        poly_path = poly_paths[i_poly]
        
        # get the Basemap coordinates of each segment
        coords_cc = np.array(
            [(vertex[0],vertex[1]) 
             for (vertex,code) in poly_path.iter_segments(simplify=False)]
        )
        
        # convert coordinates to lon/lat by 'inverting' the Basemap projection
        lon_cc, lat_cc = m(coords_cc[:,0],coords_cc[:,1], inverse=True)
    
        
        lats.extend(lat_cc.tolist()+[None]) 
        lons.extend(lon_cc.tolist()+[None])
        
       
    return lons, lats


# Function generating coastline lon/lat 
def get_coastline_traces():
    poly_paths = m.drawcoastlines().get_paths() # coastline polygon paths
    N_poly = 91  # use only the 91st biggest coastlines (i.e. no rivers)
    cc_lons, cc_lats= polygons_to_traces(poly_paths, N_poly)
    return cc_lons, cc_lats

# Function generating country lon/lat 
def get_country_traces():
    poly_paths = m.drawcountries().get_paths() # country polygon paths
    N_poly = len(poly_paths)  # use all countries
    country_lons, country_lats= polygons_to_traces(poly_paths, N_poly)
    return country_lons, country_lats

def degree2radians(degree):
    #convert degrees to radians
    return degree*pi/180

def mapping_map_to_sphere(lon, lat, radius=1):
    #this function maps the points of coords (lon, lat) to points onto the  sphere of radius radius
    
    lon=np.array(lon, dtype=np.float64)
    lat=np.array(lat, dtype=np.float64)
    lon=degree2radians(lon)
    lat=degree2radians(lat)
    xs=radius*cos(lon)*cos(lat)
    ys=radius*sin(lon)*cos(lat)
    zs=radius*sin(lat)
    return xs, ys, zs


def get_globe(lons, lats):
    # Get list of of coastline, country, and state lon/lat 

    cc_lons, cc_lats=get_coastline_traces()
    country_lons, country_lats=get_country_traces()

    #concatenate the lon/lat for coastlines and country boundaries:
    lons=cc_lons+[None]+country_lons
    lats=cc_lats+[None]+country_lats

    xs, ys, zs=mapping_map_to_sphere(lons, lats, radius=1.01)# here the radius is slightly greater than 1 
                                                         #to ensure lines visibility; otherwise (with radius=1)
                                                         # some lines are hidden by contours colors

    boundaries=dict(type='scatter3d',
               x=xs,
               y=ys,
               z=zs,
               mode='lines',
               line=dict(color='black', width=1)
              )

    colorscale=[[0.0, '#313695'],
                [0.07692307692307693, '#3a67af'],
                [0.15384615384615385, '#5994c5'],
                [0.23076923076923078, '#84bbd8'],
                [0.3076923076923077, '#afdbea'],
                [0.38461538461538464, '#d8eff5'],
                [0.46153846153846156, '#d6ffe1'],
                [0.5384615384615384, '#fef4ac'],
                [0.6153846153846154, '#fed987'],
                [0.6923076923076923, '#fdb264'],
                [0.7692307692307693, '#f78249'],
                [0.8461538461538461, '#e75435'],
                [0.9230769230769231, '#cc2727'],
                [1.0, '#a50026']]

    clons=np.array(lon.tolist()+[180], dtype=np.float64)
    clats=np.array(lat, dtype=np.float64)
    clons, clats=np.meshgrid(clons, clats)

    XS, YS, ZS=mapping_map_to_sphere(clons, clats)


    sphere=dict(type='surface',
                x=XS, 
                y=YS, 
                z=ZS,
                colorscale=colorscale,
                surfacecolor=OLR,
                cmin=-20, 
                cmax=20,
                colorbar=dict(thickness=20, len=0.75, ticklen=4, title= 'W/m²')
                )

    noaxis=dict(showbackground=False,
                showgrid=False,
                showline=False,
                showticklabels=False,
                ticks='',
                title='',
                zeroline=False)

    layout3d=dict(title='Outgoing Longwave Radiation Anomalies<br>Dec 2017-Jan 2018',
                font=dict(family='Balto', size=14),
                width=800, 
                height=800,
                scene=dict(xaxis=noaxis, 
                            yaxis=noaxis, 
                            zaxis=noaxis,
                            aspectratio=dict(x=1,
                                            y=1,
                                            z=1),
                            camera=dict(eye=dict(x=1.15, 
                                        y=1.15, 
                                        z=1.15)
                                        )
                ),
                paper_bgcolor='rgba(235,235,235, 0.9)'  
            )
                

