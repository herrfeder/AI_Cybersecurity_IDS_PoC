import plotly.graph_objects as go
from plotly.subplots import make_subplots
from statsmodels.tsa.seasonal import seasonal_decompose
from pandas.tseries.offsets import DateOffset
import pandas as pd
import numpy as np
import dash_table
from ipdb import set_trace


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


def get_data_table_2(df, fig="", title="", dash=False):

    df = pd.read_csv('https://raw.githubusercontent.com/plotly/datasets/master/2014_usa_states.csv')
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
        page_size= 10,
    ),

    set_trace()
   
    if dash:
        return apply_layout(dt, title)
    else:
        fig.show()


if __name__ == "__main__":

    get_data_table_2("", fig="", title="", dash=True)