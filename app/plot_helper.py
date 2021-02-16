import plotly.graph_objects as go
from plotly.subplots import make_subplots
from statsmodels.tsa.seasonal import seasonal_decompose
from pandas.tseries.offsets import DateOffset
import pandas as pd
import numpy as np
import dash_table
import matplotlib.pyplot as plt

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
    fig = go.Figure([go.Bar(x=number, y=ips, orientation='h',
                            marker_color="#ff0000", width=0.5)])

    if dash:
        return apply_layout(fig, height=400, width=400)
    else:
        fig.show()


def pred_rf_style(df):

    df['Status_RF'] = df['Prediction_rf'].apply(lambda x:
                                                '🎄😎' if x < 0.6 else (
                                                    '🎄😨' if x < 0.9 else '🎄😱'
                                                ))

    df['Status_NN'] = df['Prediction_nn'].apply(lambda x:
                                                '❄️😎' if x < 0.6 else (
                                                    '❄️😨' if x < 0.9 else '❄️😱'
                                                ))

    df['Status_AD'] = df['Prediction_AD'].apply(lambda x:
                                                '🚨😱' if x == 1 else '🚨😎'
                                                )

    return df


def plot_prediction_table(df, fig="", title=""):

    df = pred_rf_style(df)

    plot_df = df[["Status_RF", "Status_NN", "Status_AD", "Time", "id.orig_h", "id.orig_p",
                  "id.resp_h", "id.resp_p", "proto"]]

    dt = dash_table.DataTable(
        id='datatable-row-ids',
        columns=[
            {'name': i, 'id': i, 'deletable': True} for i in plot_df.columns
            # omit the id column
            if i != 'id'
        ],
        data=plot_df.to_dict('records'),
        editable=False,
        sort_action="native",
        sort_mode='multi',
        row_selectable='multi',
        row_deletable=False,
        selected_rows=[],
        page_action='native',
        page_current=0,
        page_size=100,
        style_as_list_view=True,
        style_data_conditional=[
            {
                'if': {'column_id': 'Status_RF'},
                'fontSize': '2rem',
            },
            {
                'if': {'column_id': 'Status_NN'},
                'fontSize': '2rem',
            },
            {
                'if': {'column_id': 'Status_AD'},
                'fontSize': '2rem',
            },
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
        style_cell={'border': '10px solid #222', 'height': '50px'},
        style_table={'height': '1000px', 'width': '2000px',
                     'overflowY': 'auto',
                     'overflowX': 'auto'
                     }
    )

    return dt


def plot_data_table(df, fig="", title=""):

    plot_df = df[["Time", "id.orig_h", "id.orig_p",
                  "id.resp_h", "id.resp_p", "proto",
                  "service", "orig_ip_bytes", "resp_ip_bytes",
                  "duration", "orig_bytes", "resp_bytes"]]

    dt = dash_table.DataTable(
        id='datatable-row-ids',
        columns=[
            {'name': i, 'id': i, 'deletable': True} for i in plot_df.columns
            # omit the id column
            if i != 'id'
        ],
        data=plot_df.to_dict('records'),
        editable=False,
        sort_action="native",
        sort_mode='multi',
        row_selectable='multi',
        row_deletable=False,
        selected_rows=[],
        page_action='native',
        page_current=0,
        page_size=50,
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
        style_cell={'border': '10px solid #222'},
        style_table={'height': '400px', 'width': '1000px',
                     'overflowY': 'auto',
                     'overflowX': 'auto'
                     }
    )

    return dt


def plot_monitor_scatter(df, fig="", title="", dash=False):

    resp_pkts = df.groupby(pd.Grouper(freq='1Min', base=30, label='right'))[
        "resp_pkts"].sum()
    orig_pkts = df.groupby(pd.Grouper(freq='1Min', base=30, label='right'))[
        "orig_pkts"].sum()
    new_index = df.groupby(pd.Grouper(freq='1Min', base=30, label='right'))[
        "orig_pkts"].sum().index

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


def matplotlib_to_plotly(cmap, pl_entries):
    h = 1.0 / (pl_entries - 1)
    pl_colorscale = []

    for k in range(pl_entries):
        C = list(map(np.uint8, np.array(cmap(k * h)[:3]) * 255))
        pl_colorscale.append([k * h, 'rgb' + str((C[0], C[1], C[2]))])

    return pl_colorscale


def plot_anomaly(X_train, xx, yy, Z, dash=True):

    back = go.Contour(x=xx,
                      y=yy,
                      z=Z,
                      colorscale=matplotlib_to_plotly(plt.cm.Blues_r, len(Z)),
                      showscale=False,
                      line=dict(width=0)
                      )

    b1 = go.Scatter(x=X_train[:, 0],
                    y=X_train[:, 1],
                    name="training observations",
                    mode='markers',
                    marker=dict(color='white', size=7,
                                line=dict(color='black', width=1))
                    )

    layout = go.Layout(title="IsolationForest",
                       hovermode='closest')
    data = [back, b1]

    fig = go.Figure(data=data, layout=layout)

    if dash:
        return apply_layout(fig, title="", height=800, width=1000)
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
                         color=number,
                         projection="orthographic")

    fig.layout.geo["bgcolor"] = "rgb(34,34,34)"

    if dash:
        return apply_layout(fig, title, 400)
    else:
        fig.show()
