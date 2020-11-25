import plotly.graph_objects as go
from plotly.subplots import make_subplots
from statsmodels.tsa.seasonal import seasonal_decompose
from pandas.tseries.offsets import DateOffset
import pandas as pd
import numpy as np
import dash_table
from ipdb import set_trace

import numpy as np      
import plotly.express as px     





def apply_layout(fig, title="", height=1250, width=600):
    '''
    Applying web app layout for plots.
    
    INPUT:
        title - (str) Figure Title
        height - (int) Figure Height 
    OUTPUT:
        fig - (plotly Figure) plot object that needs to be passed to dash figure attribute
    '''
    
    
    fig.update_layout(height=height, width=width, title_text=title)
    layout = fig["layout"]
    layout["paper_bgcolor"] = "rgba(0,0,0,0)"   
    layout["plot_bgcolor"] = "rgba(0,0,0,0)"
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

    ips = data_dict["ips"]
    number = data_dict["number"]
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


def plot_data_table(df, fig="", title=""):

    plot_df = df[["id.orig_h", "id.orig_p", 
                  "id.resp_h","id.resp_p","proto",
                  "service", "orig_ip_bytes","resp_ip_bytes",
                  "duration","orig_bytes", "resp_bytes"]]

    dt = dash_table.DataTable(
        id='datatable-row-ids',
        columns=[
            {'name': i, 'id': i, 'deletable': True} for i in plot_df.columns
            # omit the id column
            if i != 'id'
            ],
        data=plot_df.to_dict('records'),
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
        style_table= {'height': '400px', 'width': '1000px',
                      'overflowY': 'auto',
                      'overflowX': 'auto'
                     }
    )

    
   
    return dt



def plot_monitor_scatter(df, fig="", title="", dash=False):

    resp_pkts = df.groupby(pd.Grouper(freq='1Min', base=30, label='right'))["resp_pkts"].sum()
    orig_pkts = df.groupby(pd.Grouper(freq='1Min', base=30, label='right'))["orig_pkts"].sum()
    new_index = df.groupby(pd.Grouper(freq='1Min', base=30, label='right'))["orig_pkts"].sum().index

    if not fig:
        fig = make_subplots(
                        rows=1, 
                        cols=1, 
                        shared_xaxes=True)
     
        fig.add_trace(go.Scatter(x=new_index, 
                             y=resp_pkts,
                             line=dict(color='#2cfec1'),
                             name="Response Packets"), row=1, col=1)

        fig.add_trace(go.Scatter(x=new_index, 
                             y=orig_pkts,
                             line=dict(color='#2ac9fe'),
                             name="Origin Packets"), row=1, col=1)

    if dash:
        return apply_layout(fig, title, height=400, width=800)
    else:
        fig.show()




def get_world_plot(data_dict, fig="", title="", dash=False):

    ips = data_dict["ips"]
    number = data_dict["number"]
    lonlat = data_dict["lonlat"]

    long_list = [i[0] for i in lonlat]
    lat_list = [i[1] for i in lonlat]

    fig = px.scatter_geo(lon=long_list, lat=lat_list,
                     size=number,
                     color =number,
                     projection="orthographic")
    
    fig.layout.geo["bgcolor"] = "rgb(34,34,34)"

    if dash:
        return apply_layout(fig, title, 400)
    else:
        fig.show()