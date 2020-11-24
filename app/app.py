import json
import base64
import datetime
import requests
import pathlib
import math
import pandas as pd
import flask
import dash
import dash_table
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
import plotly.graph_objs as go
import app.plot_helper as ph
import app.data_helper as dh

from dash.dependencies import Input, Output, State
from plotly import tools


# set base app directory path
APP_PATH = pathlib.Path(__file__).parent.resolve()


dh = dh.IDSData()
dh.read_source("conn")
dh.update_source("conn")



# Lists for Project Menu    
acc_str_list = ["STATIC ANALYSIS",
                "MONITOR TRAFFIC",
                "CRAWL N TRAIN", 
                "APPLY MODEL"
                ]

svg_icon_src = ["https://raw.githubusercontent.com/herrfeder/herrfeder.github.io/master/daten-kreisdiagramm.svg",
                "https://raw.githubusercontent.com/herrfeder/herrfeder.github.io/master/monitor.svg",
                "https://raw.githubusercontent.com/herrfeder/herrfeder.github.io/master/164053.svg",
                "https://raw.githubusercontent.com/herrfeder/herrfeder.github.io/master/cloud-computing.svg"]





### Assistance Functions for creating HTML content ###

def make_items(acc_str_list, svg_icon_src):
    '''
    Populates Bootstrap accordion Menu with sub menu sliders from given input arguments.
    The given DOM id's are crucial for app callbacks, therefore shouldn't be modified.
    
    INPUT:
        acc_str - (list) List of strings with Top Menu Bullets
        acc_slider_list - (list of lists) List of list with strings for sub Menu Slider
    OUTPUT:
        card_list - (list of dash HTML objects) Will hold all menu points and submenu slider 
    '''
    card_list = []
    for acc_str, svg_icon in zip(acc_str_list, svg_icon_src):
        card_list.append(html.Div(
            id=f"menuitem-{acc_str}",
            children=[
                dbc.CardHeader([
                    dbc.Row([
                        html.Span(id=f"spandot-{acc_str}",
                                    className="menuicon-inactive",
                                    children=[
                                        html.Span(id=f"spandoti-{acc_str}",
                                            style={    
                                                "height": "40px",
                                                "width": "40px",
                                                "background-image": f"url({svg_icon})",
                                                "background-repeat": "no-repeat",
                                                "display": "grid"
                                                }
                                                )]
                                        
                                    ),
                    dbc.Button(
                        f"{acc_str}",
                        id=f"button-{acc_str}",
                        color="link",
                        style={"padding-top":10,
                               "font-color":"orange",
                            "align": "center"}

                        )
                    ], className="menurow", 
                    id=f"row-{acc_str}",
                    style={"display":"inline-flex", 
                                "align-items":"center",
                                "padding-left":5,
                                "padding-right":20} 
                    )
                ], className="menucard", 
                id=f"menucard-{acc_str}"
        )
    ]))
    
         
    return card_list

                    
### BASIC WEB APP LAYOUT ###

# NAVBAR

NAVBAR = dbc.Navbar(
    children=[
       
            dbc.Row(
                [
                    dbc.Col(html.A(html.Img(src="https://upload.wikimedia.org/wikipedia/commons/2/21/BWI_GmbH_logo.svg", height="40px"), href="https://www.udacity.com"), width=1),
                    dbc.Col(dbc.NavbarBrand(dbc.Row([html.P("BWI Datalytics Hackathon 2020 ►", style={"color":"#ff0000"}),
                                                     html.P("BroAI", style={"color":"orange"}),
                                                     html.P("(KI - Cyber Security)", style={"color":"grey"})], align="center")), width=9),
                                                     
                    dbc.Col(dbc.DropdownMenu(
                        children=[
                            dbc.DropdownMenuItem("blubb",
                                        href="blubb"),
                            dbc.DropdownMenuItem("blubb", 
                                        href="blubb"),
                          
                        ],
                        nav=False,
                        in_navbar=True,
                        label="by Team NastyNULL",
                        style={"color": "white", "font-size": 10, "font-weight":"lighter"},
                    ), width=2),
                    

                ],
                align="center",
                no_gutters=True,
                style={"width":"100%"}
            ),
           
        
    ],
    color="dark",
    dark=True,
    sticky="top",
)


# Basic Web App Skeleton

LEFT_COLUMN = dbc.Jumbotron(
    [
        html.Div(make_items(acc_str_list, svg_icon_src), className="accordion"),   
    ], style={"padding": "0rem 1rem 1rem 1rem"}
)

RIGHT_COLUMN = html.Div(id="right_column", children=[html.Div(id="right_column_loading")])


BODY = dbc.Container([
            dbc.Row(
                [
                    dbc.Col(LEFT_COLUMN, md=1),
                    dbc.Col(RIGHT_COLUMN, md=11),
                ],
                style={"marginTop": 20},
            ),
     
            ], fluid=True)


### Prepared Visualisations for speeding up web app ###

def return_ip_bar_chart(timespan=""):
    data_dict = dh.get_ten_most_source_ip(timespan)
    
    return ph.plot_ten_most_ip(data_dict, title="",dash=True)
    

 
VIEW_ZEEK_TABLE =  ph.get_data_table_2(dh.df_d["conn"].tail(50), fig="", title="")
WORLD_MAP = ""


timespan_options = ["15min","30min","1h","5h","12h","24h"]

timespan_list = []
for num, option in enumerate(timespan_options):
    timespan_list.append({"label": option,
                          "value": num})


MONITOR_TIME_DROPDOWN = html.Div([
                        dcc.Dropdown(id='time_dropdown',
                                     options=timespan_list,
                                     value=1)

                     ,], style={"width":"100%"})

MONITOR_TIME_LABEL = html.Label("Timespan:",
                         style={"padding-left":5,
                                "padding": 10}) 

### MENU BASED CONTENT ###

# STATIC ANALYSIS

# MONITOR TRAFFIC

EXP_CHART_PLOT = [dbc.Row(children=[
                    dbc.Col([dbc.CardHeader(html.H5("Controls")), dbc.CardBody([MONITOR_TIME_LABEL,MONITOR_TIME_DROPDOWN])], md=1),
                    dbc.Col([dbc.CardHeader(html.H5("Most frequently Source IPs")), dbc.CardBody(dcc.Loading(dcc.Graph(figure=return_ip_bar_chart(), id="most_ip_plot")))]),
                    dbc.Col([dbc.CardHeader(html.H5("Second Plot")),dbc.CardBody()])
                 ]),
                  dbc.Row(children=[
                    dbc.Col([dbc.CardHeader(html.H5("BRO List")), dbc.CardBody(html.Div(children=[VIEW_ZEEK_TABLE], id="monitor_data_table"))])  
                  ]),
                  dcc.Interval(id='table_update', interval=1*5000, n_intervals=0)
                 ]


# CRAWLIN N TRAIN

# APPLY MODEL


### WEBAPP INIT ###

app = dash.Dash(__name__, 
                external_stylesheets=[dbc.themes.DARKLY], 
                url_base_pathname="/aicspoc/",
                meta_tags=[
                    {"name": "viewport", "content": "width=device-width, initial-scale=1.0"}
                ],
               )

app.layout = html.Div(children=[NAVBAR, BODY])
app.config['suppress_callback_exceptions'] = True
app.css.config.serve_locally = True
app.scripts.config.serve_locally = True

server = app.server


### CALLBACKS ###

@app.callback([Output('monitor_data_table', 'children'),
               Output('most_ip_plot', 'figure')],
              [Input('table_update', 'n_intervals')])
def update_table_data(n_intervals):
    dh.update_source("conn")
    return ph.get_data_table_2(dh.df_d["conn"].tail(50), fig="", title=""), return_ip_bar_chart(timespan="")



# Menu control function
acc_input = [Input(f"menuitem-{i}", "n_clicks") for i in acc_str_list]
@app.callback(
    Output("right_column_loading", "children"),
    acc_input,
    [State("right_column_loading", "children")],
)    
def show_plot(acc_01, acc_02, acc_03, acc_04, right_children):
    '''
    This function returns the HTML content into the right column of web app based
    on the clicked accordion button and clicked submenu slider value
    '''
    ctx = dash.callback_context

    if not ctx.triggered:
        return ""
    else:
        element_id = ctx.triggered[0]["prop_id"].split(".")[0]

    if (acc_str_list[0] in element_id):
        return ""
        
    elif (acc_str_list[1] in element_id):
        return EXP_CHART_PLOT
        
    elif (acc_str_list[2] in element_id):
        return ""
        
    elif (acc_str_list[3] in element_id):
        return ""
    else:
        return ""
        
    

            
# controls green color of menu dot    
@app.callback(
    [Output(f"spandot-{i}", "className") for i in acc_str_list],
    [Input(f"menuitem-{i}", "n_clicks") for i in acc_str_list],
    [State(f"spandot-{i}", "className") for i in acc_str_list],
)
def toggle_active_dot(n1, n2, n3, n4,
                      active1, active2, active3, active4):
    '''
    Based on click events on the accordion button the style of the spandot in each accordion button
    will be updated
    '''
    
    sty_a = "menuicon-active"
    sty_na = "menuicon-inactive"
    
    ctx = dash.callback_context

    if not ctx.triggered:
        return sty_na, sty_na, sty_na, sty_na
    else:
        button_id = ctx.triggered[0]["prop_id"].split(".")[0]
    if (acc_str_list[0] in button_id):
        return sty_a, sty_na, sty_na, sty_na
    elif (acc_str_list[1] in button_id):
        return sty_na, sty_a, sty_na, sty_na
    elif (acc_str_list[2] in button_id):
        return sty_na, sty_na, sty_a, sty_na
    elif (acc_str_list[3] in button_id):
        return sty_na, sty_na, sty_na, sty_a
    else:
        return sty_na, sty_na, sty_na, sty_na


    
if __name__ == "__main__":
    app.run_server(debug=True,  port=8050, host="0.0.0.0")
