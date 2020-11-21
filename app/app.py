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
import app.data_prep_helper as dh

from dash.dependencies import Input, Output, State
from plotly import tools


# set base app directory path
APP_PATH = pathlib.Path(__file__).parent.resolve()

# prepare the DataSet with all Input Time Series
do_big = dh.ModelData(chart_col=["Price", "High", "Low", "Price_norm"])

# smaller subset of columns for some visualisation use cases
small_col_set = ['bitcoin_Price', 'sp500_Price', 'dax_Price', 'googl_Price',
                 'gold_Price', 'alibaba_Price', 'amazon_Price', 'bitcoin_Google_Trends',
                 'cryptocurrency_Google_Trends', 'trading_Google_Trends',
                 'bitcoin_pos_sents', 'bitcoin_neg_sents', 'bitcoin_quot_sents',
                 'economy_pos_sents', 'economy_neg_sents', 'economy_quot_sents']

# columns for drop dowwn
BASE_COLUMNS = list(do_big.chart_df[small_col_set])

column_small_labels = []
for col in BASE_COLUMNS:
    column_small_labels.append({"label": col,
                                "value": col,
                                })

# dates for prediction range
FORE_DAYS = do_big.get_forecast_dates()

fore_days_labels = []
for day in FORE_DAYS:
    fore_days_labels.append({"label": day,
                             "value": day})


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


### Prepared Visualisations for speeding up web app ###

 
VIEW_DATA_FIG =  ph.get_data_table_2("", fig="", title="")




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
                                "padding-left":20,
                                "padding-right":20} 
                    )
                ], className="menucard", 
                id=f"menucard-{acc_str}"
        )
    ]))
    
         
    return card_list



def build_growth_content(growth_dict, cols):
    '''
    Populates table like HTML Elements from input dict. Used for seeing mean growth in percent of specific time series over time.
    Will color entry red for negative growth and green for positive growth.
    
    INPUT:
        growth_dict - ({str:float}) Holds column name as key and growth index as value
        cols - (list of str) Holds the column names that should be populated from input dict
    OUTPUT:
        dbc.Row - (dash HTML object) Holds the table like HTML elements
    '''
    
    col_col = []
    val_col = []
    
    for col in cols:
        val = growth_dict[col]
        if val >= 0:
            color="green"
        else:
            color="red"
          
        col_name = col.split("_")[0]
        col_col.append(dbc.Row(html.P(col_name,style={"color":color}),style={"height":"25px"}))
        val_col.append(dbc.Row(html.P(str(val)+"%",style={"color":color, "font-weight":"bold"}), style={"height":"25px"}))
    
    left_col = dbc.Col(col_col, width=8)
    right_col = dbc.Col(val_col, width=4)
    
    return dbc.Row(children=[left_col, right_col])
                    
                    
### BASIC WEB APP LAYOUT ###

# NAVBAR

NAVBAR = dbc.Navbar(
    children=[
       
            dbc.Row(
                [
                    dbc.Col(html.A(html.Img(src="https://upload.wikimedia.org/wikipedia/commons/2/21/BWI_GmbH_logo.svg", height="40px"), href="https://www.udacity.com"), width=1),
                    dbc.Col(dbc.NavbarBrand(dbc.Row([html.P("BWI Datalytics Hackathon 2020 ►", style={"color":"#ff0000"}),
                                                     html.P("BroAI", style={"color":"orange"}),
                                                     html.P("%20(KI - Cyber Security)", style={"color":"grey"})], align="center")), width=9),
                                                     
                    dbc.Col(dbc.DropdownMenu(
                        children=[
                            dbc.DropdownMenuItem("LinkedIn",
                                        href="https://www.linkedin.com/in/davidlassig/"),
                            dbc.DropdownMenuItem("Github Repo", 
                                        href="https://github.com/herrfeder/Udacity-Data-Scientist-Capstone-Multivariate-Timeseries-Prediction-Webapp.git"),
                          
                        ],
                        nav=False,
                        in_navbar=True,
                        label="by David Lassig",
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


### MENU BASED CONTENT ###

# STATIC ANALYSIS

# MONITOR TRAFFIC

EXP_CHART_PLOT = [dbc.CardHeader(html.H5("BRO List")),
                  dbc.CardBody(html.Div(children=[ 
                        html.Div(dcc.Loading(VIEW_DATA_FIG,id="data_table"))
                        ]))
                 ]

VIEW_CONCLUSIONS = html.Div(dcc.Markdown(conclusion_texts.view_data_conclusion), id="resources")


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
        return EXP_CHART_PLOT
        
    elif (acc_str_list[1] in element_id):
        return ""
        
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
        print("not_triggered")
    else:
        button_id = ctx.triggered[0]["prop_id"].split(".")[0]
        print(button_id)

    if (acc_str_list[0] in button_id):
        print("button 1 pressed")
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
