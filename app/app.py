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


global monitor_time_interval
monitor_time_interval = 60

# set base app directory path
APP_PATH = pathlib.Path(__file__).parent.resolve()



dh = dh.IDSData()
dh.read_source("conn")
dh.update_source("conn")



# Lists for Project Menu and associated Icons    
acc_str_list = ["CONCEPT",
                "MONITOR TRAFFIC",
                "CRAWL N TRAIN", 
                "APPLY MODEL"
                ]

svg_icon_src = ["https://raw.githubusercontent.com/herrfeder/herrfeder.github.io/master/71438.svg",
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
                    dbc.Col(html.A(html.Img(src="https://abload.de/img/bwi_dataanalyticshack7ujy4.png", height="40px"), href="https://www.bwi.de"), width=2),
                    dbc.Col(dbc.NavbarBrand(dbc.Row([
                                                     html.P("BroAI", style={"color":"#FF0000"}),
                                                     html.P("(KI - Cyber Security)", style={"color":"orange"})], align="center")), width=7),
                                                     
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
    style={"background-image": "url('https://raw.githubusercontent.com/herrfeder/herrfeder.github.io/master/image.png')"}
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


### Functions where Plothelper and Datahelper comes together ###

def return_ip_bar_chart(file_type="", timespan=""):
    data_dict = dh.get_ten_most_source_ip(file_type, timespan)
    
    return ph.plot_ten_most_ip(data_dict, title="",dash=True)

def return_ip_bar_dest_chart(file_type="", timespan=""):
    data_dict = dh.get_ten_most_dest_ip(file_type, timespan)
    
    return ph.plot_ten_most_ip(data_dict, title="",dash=True)


def return_data_table(file_type="", timespan=""):
    df = dh.get_timespan_df(file_type, timespan*60)

    return ph.plot_data_table(df.tail(50), fig="", title="")


def return_scatter(file_type="", timespan=""):
    df = dh.get_timespan_df(file_type, timespan*60)

    return ph.plot_monitor_scatter(df, title="", dash=True)


def return_world(file_type="", timespan=""):
    data_dict = dh.get_longitude_latitude(file_type, timespan)
    return ph.get_world_plot(data_dict, dash=True)


def return_apply_table():
    df = dh.get_timespan_df("conn", dh.anomaly_detection_counter+600)

    pred = dh.return_anomaly_prediction(df)
    
    plot_df = df.tail(99)
    print(plot_df.shape)
    plot_df["Prediction_AD"] = pred
    return ph.plot_prediction_table(plot_df, fig="", title="")



def return_anomaly_model(file_type="", train_offset="", counter_offset=""):

    X_train, xx, yy, Z = dh.train_anomaly_detection(file_type, train_offset, counter_offset)
    return ph.plot_anomaly(X_train, xx, yy, Z)




 
WORLD_MAP = ""
MONITOR_SCATTER = ""


timespan_labels = ["15min","30min","1h","5h","12h","24h"]
timespan_values = [15, 30, 60, 300, 720, 1440]

timespan_list = []
for label, value in zip(timespan_labels, timespan_values):
    timespan_list.append({"label": label,
                          "value": value})


anomaly_counter_labels = ["5min","10min","15min","30min", "1h"]
anomaly_counter_values = [5*60, 10*60, 15*60, 30*60, 60*60]

anomaly_counter_list = []
for label, value in zip(anomaly_counter_labels, anomaly_counter_values):
    anomaly_counter_list.append({"label": label,
                                 "value": value})



anomaly_span_labels = ["3h","6h","12h","24h","48h", "72h"]
anomaly_span_values = [3, 6, 12, 24, 48, 72]


anomaly_span_list = []
for label, value in zip(anomaly_span_labels, anomaly_span_values):
    anomaly_span_list.append({"label": label,
                          "value": value})


MONITOR_TIME_DROPDOWN = html.Div([
                        dcc.Dropdown(id='monitor_time_dropdown',
                                     options=timespan_list,
                                     value=60)

                     ,], style={"width":"100%", "color": "black"})

MONITOR_TIME_LABEL = html.Label("Timespan:",
                         style={"padding-left":5,
                                "padding": 10}) 


ANOMALY_COUNTER_DROPDOWN = html.Div([
                        dcc.Dropdown(id='anomaly_counter_dropdown',
                                     options=anomaly_counter_list,
                                     value=5*60)
                                     ,], style={"width":"100%", "color": "black"})


ANOMALY_SPAN_DROPDOWN = html.Div([
                        dcc.Dropdown(id='anomaly_span_dropdown',
                                     options=anomaly_span_list,
                                     value=72)
                        ,], style={"width":"100%", "color": "black"})

ANOMALY_LABEL = html.Label("Choose Anomaly Model:",
                         style={"padding-left":5,
                                "padding": 10}) 


### MENU BASED CONTENT ###

# CONCEPT

CONCEPT = html.Img(src="https://raw.githubusercontent.com/herrfeder/herrfeder.github.io/master/pitch_final.png", style={"width":"100%"})

# MONITOR TRAFFIC

MONITOR_FRONTEND = [dbc.Row(children=[
                    dbc.Col([dbc.CardHeader(html.H5("Controls")), dbc.CardBody([MONITOR_TIME_LABEL,MONITOR_TIME_DROPDOWN])], md=1),
                    dbc.Col([dbc.CardHeader(html.H5("Most frequently Source IPs")), 
                             dbc.CardBody(dcc.Loading(dcc.Graph(figure="", id="most_ip_plot"),color="#FF0000"))
                    ], md=3),
                    dbc.Col([dbc.CardHeader(html.H5("Most frequently Destination IPs")), 
                             dbc.CardBody(dcc.Loading(dcc.Graph(figure="", id="most_ip_dest_plot"),color="#FF0000"))
                    ],md=3),
                    dbc.Col([dbc.CardHeader(html.H5("Location of Source IPs")),
                             dbc.CardBody(dcc.Loading(dcc.Graph(figure="", id="world_map_plot"),color="#FF0000"))])
                 ]),
                  dbc.Row(children=[
                    dbc.Col([dbc.CardHeader(html.H5("Connection List")), dbc.CardBody(html.Div(children=[], id="monitor_data_table"))]),
                    dbc.Col([dbc.CardHeader(html.H5("Connection over Time")), 
                             dbc.CardBody(dcc.Loading(dcc.Graph(figure="", id="monitor_scatter_plot"), color="#FF0000"))])    
                  ]),
                    dcc.Interval(id='table_update', interval=1*10000, n_intervals=0)
                 ]


# CRAWLIN N TRAIN

roccurve = """
![](https://raw.githubusercontent.com/herrfeder/herrfeder.github.io/master/Selection_006.png)
"""

roccurve_zoom = """
![](https://raw.githubusercontent.com/herrfeder/herrfeder.github.io/master/Selection_007.png)
"""

nn_confusion = """
![](https://raw.githubusercontent.com/herrfeder/herrfeder.github.io/master/nn_confusion_matrix_v1.png)
"""

rf_confusion = """
![](https://raw.githubusercontent.com/herrfeder/herrfeder.github.io/master/random_forest_confusion_matrix_v2.png)
"""



tab1_content = [dbc.Row(children=[
                    dbc.Col([dbc.CardHeader(html.H5("ROC Curves")), dbc.CardBody(dcc.Markdown(roccurve, className="image_big"))], md=6),
                    dbc.Col([dbc.CardHeader(html.H5("ROC Curves Zoom")), dbc.CardBody(dcc.Markdown(roccurve_zoom, className="image_big"))], md=6),]),
                dbc.Row(children=[
                    dbc.Col([dbc.CardHeader(html.H5("Random Forest Confusion Matrix")), dbc.CardBody(dcc.Markdown(rf_confusion))], md=6),
                    dbc.Col([dbc.CardHeader(html.H5("Neuronal Network Confusion Matrix")), dbc.CardBody(dcc.Markdown(nn_confusion))], md=6),]),
                ]


tab2_content = dbc.Row(children=[
    dbc.Card( 
        dbc.CardBody(
        [  dbc.Col( children=[ html.Label("Anomaly Counter Range:", style={"padding-left":5, "padding": 10}),
                               ANOMALY_COUNTER_DROPDOWN,
                               html.Label("Anomaly Span Range:", style={"padding-left":5, "padding": 10}),
                               ANOMALY_SPAN_DROPDOWN,
                                dbc.Button("Train Anomaly Detection Model", id="anomaly_submit",color="success")]),
           

           dbc.Col(children=[dcc.Loading(dcc.Graph(id="anomaly_result"), color="#FF0000", className="loading_anomaly")]) ]

        
    ),
    style={"padding-top":"20px","width":"80%"}

)])


TRAINING = dbc.Tabs(
    [
        dbc.Tab(tab1_content, label="Random Forest & Neural Net"),
        dbc.Tab(tab2_content, label="Anomaly Detection"),
        
    ])


# APPLY MODEL

APPLY_FRONTEND = [dbc.Row(children=[html.Div(children=[], id="apply_dummy"),
                    dbc.Col([dbc.CardHeader(html.H5("Predict Attacks")), dbc.CardBody(html.Div(children=[], id="apply_data_table"))
                    ]),
                    ]),
                    dcc.Interval(id='apply_update', interval=1*5000, n_intervals=0),
                ]



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


# create anomaly model and output visualisation of isolation forest
@app.callback(Output('anomaly_result', 'figure'),
               [Input('anomaly_counter_dropdown', 'value'),
                Input('anomaly_span_dropdown', 'value'),
                Input('anomaly_submit', 'n_clicks')])
def return_anomaly(counter_value, span_value, anomaly_click):
    dh.update_source("conn")

    return return_anomaly_model(file_type="conn", train_offset=span_value, counter_offset=counter_value)

# output table with predictions
@app.callback(Output('apply_data_table', 'children'),
               [Input('apply_update', 'n_intervals'),])
def update_apply_data(n_intervals):
    dh.update_source("conn")

    return return_apply_table()
         

# update output table
@app.callback([Output('monitor_data_table', 'children'),
               Output('most_ip_plot', 'figure'),
                Output('most_ip_dest_plot', 'figure'),
               Output('monitor_scatter_plot', 'figure'),
               Output('world_map_plot', 'figure')],
               [Input('table_update', 'n_intervals'),
                Input('monitor_time_dropdown', 'value')])
def update_monitor_table(n_intervals, monitor_time_interval):
    dh.update_source("conn")
    return (return_data_table("conn", timespan=monitor_time_interval), 
           return_ip_bar_chart("conn", timespan=monitor_time_interval),
           return_ip_bar_dest_chart("conn", timespan=monitor_time_interval),
           return_scatter("conn", timespan=monitor_time_interval),
           return_world("conn", timespan=monitor_time_interval))


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
        return CONCEPT
    else:
        element_id = ctx.triggered[0]["prop_id"].split(".")[0]

    if (acc_str_list[0] in element_id):
        return CONCEPT
        
    elif (acc_str_list[1] in element_id):
        return MONITOR_FRONTEND
        
    elif (acc_str_list[2] in element_id):
        return TRAINING
        
    elif (acc_str_list[3] in element_id):
        return APPLY_FRONTEND
    else:
        return CONCEPT
        
    

            
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
