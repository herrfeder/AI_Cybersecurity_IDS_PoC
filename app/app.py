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
import app.conclusion_texts as conclusion_texts

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
acc_str_list = ["A Introduction",
                "B View Data", 
                "C Correlation Analysis",
                "D Causality Analysis",
                "E Model Evaluation",
                "F Forecast"]

acc_slider_list = [["Introduction", "Resources"],
                   ["View Data","Conclusions"],
                   ["Simple Correlation", "Correlation Timeshift", "Conclusions"],
                   ["Seasonal Analysis", "Granger Causality", "Conclusions"],
                   ["ARIMAX", "GRU", "Conclusions"],
                   ["Forecast", "$ Buying Simulation $", "Chances and next Steps"]]
    


### Prepared Visualisations for speeding up web app ###

GRANGER_PATH = APP_PATH.joinpath("data/granger_causality.csv")
GRANGER_PLOT = ph.return_granger_plot(GRANGER_PATH, title="", colormap="viridis_r", dash=True)

VIEW_DATA_FIG = ph.exploratory_plot(do_big.apply_boll_bands("bitcoin_hist",
                                                              append_chart=False), title="", dash=True)

CORR_STATIC_FIG = ph.plot_val_heatmap(do_big.chart_df[small_col_set].corr(), 
                                          title="", 
                                          dash=True)

SEASON_PLOT = ph.return_season_plot(do_big.chart_df[small_col_set], 
                                    "bitcoin_Price", 
                                    title="", 
                                    dash=True)

CROSS_VAL_ARIMAX = ph.return_cross_val_plot(do_big.cross_validate_arimax(),
                                            title="",
                                            dash=True)

CROSS_VAL_GRU = ph.return_cross_val_plot(do_big.cross_validate_gru(),
                                         title="",
                                         dash=True)


### Assistance Functions for creating HTML content ###

def make_items(acc_str_list, acc_slider_list):
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
    for acc_str, acc_slider in zip(acc_str_list, acc_slider_list):
        card_list.append(dbc.Card(
            [
                dbc.CardHeader(
                        dbc.Row([
                            html.Span(id=f"spandot-{acc_str}",
                                      style={"height": "15px", 
                                             "width": "15px", 
                                             "background-color": "#bbb", 
                                             "border-radius": "50%",
                                             "padding-left": 20
                                             }
                                     ),
                        dbc.Button(
                            f"{acc_str}",
                            color="link",
                            id=f"group-{acc_str}-toggle",
                            style={"padding-top":10}

                            )
                        ],style={"display":"inline-flex", 
                                 "align-items":"center",
                                 "padding-left":20} 
                        ) 
                    ),


                dbc.Collapse(
                    html.Div(children=[dbc.Col(
                                dcc.Slider(
                                        id = f"slider-{acc_str}",
                                        updatemode = "drag",
                                        vertical = True,
                                        step=None,
                                        marks = {index: {"label":"{}".format(name),
                                                         "style": {"color": "#2AA198"}
                                                        } for index,name in enumerate(acc_slider[::-1])},
                                        min=0,
                                        max=len(acc_slider)-1,
                                        value=len(acc_slider)-1,
                                        verticalHeight=len(acc_slider)*50)
                            ),
                            dbc.Col(html.P(id=f"slidersub-{acc_str}", 
                                           style={"color":"orange"}), 
                                    id=f"slider-content-{acc_str}")],
                            style={"padding":10, "padding-left":20}), id=f"collapse-{acc_str}"       
                )
            ])
    )
         
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
                    dbc.Col(html.A(html.Img(src="https://upload.wikimedia.org/wikipedia/commons/3/3b/Udacity_logo.png", height="40px"), href="https://www.udacity.com"), width=1),
                    dbc.Col(dbc.NavbarBrand(dbc.Row([html.P("DataScience Nanodegree Capstone Project ►", style={"color":"#02b3e4"}),
                                                     html.P("Multivariate Timeseries Analysis (Stock Market)")], align="center")), width=9),
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
        html.H4(children="Project Menu", className="display-5"),
        html.P(
            "(Click A Field to continue)",
            style={"fontSize": 12, "font-weight": "lighter"},
        ),
        html.Hr(className="my-2"),
        html.Div(make_items(acc_str_list, acc_slider_list), className="accordion"),   
    ]
)

RIGHT_COLUMN = html.Div(id="right_column", children=[html.Div(id="right_column_loading")])


BODY = dbc.Container([
            dbc.Row(
                [
                    dbc.Col(LEFT_COLUMN, md=3),
                    dbc.Col(RIGHT_COLUMN, md=9),
                ],
                style={"marginTop": 30},
            ),
            dbc.Row(
                [
                    dbc.Col(dbc.Card(""), width=8),
                    dbc.Col(dbc.Card(""), width=4),
                    
                ]
            )], fluid=True)


### MENU BASED CONTENT ###

# A INTRODUCTION

INTRODUCTION = html.Div(dcc.Markdown(conclusion_texts.introduction), id="introduction")

RESOURCES = html.Div(dcc.Markdown(conclusion_texts.resources), id="resources")


# B VIEW DATA

EXP_CHART_PLOT = [dbc.CardHeader(html.H5("Historic Input Datasets")),
              dbc.CardBody(dcc.Loading(dcc.Graph(id="exp_chart_plot",
                           figure=VIEW_DATA_FIG)))
             ]

VIEW_CONCLUSIONS = html.Div(dcc.Markdown(conclusion_texts.view_data_conclusion), id="resources")


# C CORRELATION ANALYSIS

CORR_SHIFT_DROPDOWN = html.Div([
                        dcc.Dropdown(id='corr_shift_dropdown',
                                     options=column_small_labels,
                                     value="bitcoin_Price")
                     ,], style={"width":"20%"})

CORR_SHIFT_SLIDER = html.Div(children=[
                        dcc.Slider(
                            id='corr_shift_slider',
                            updatemode = "drag",
                            marks={day_shift: {
                                            "label": str(day_shift),
                                            "style": {"color": "#7fafdf"},
                                        }
                                        for day_shift in range(-50,5,5) },
                                min=-50,
                                max=0,
                                step=1,
                                value=-30,),
                    ],style={"width":"40%"})

CORR_01_CHART_PLOT = [dbc.CardHeader(html.H5("Correlation Matrix for all Input Time Series")),
              dbc.CardBody(dcc.Loading(dcc.Graph(id="corr_01_matrix_plot",
                           figure=CORR_STATIC_FIG)))
             ]

CORR_02_CHART_PLOT = [dbc.CardHeader(html.H5("Correlation Matrix with User-Defined Time Shift")),
                        dbc.CardBody( 
                            html.Div(children=[
                                        dbc.Row(children=[
                                            html.Label("Column to Fix:",
                                                style={"padding-left":20,
                                                       "padding": 10}), 
                                            CORR_SHIFT_DROPDOWN,
                                            html.Label("Past Timeshift:",
                                                style={"padding-left":20,
                                                       "padding": 10}),
                                            CORR_SHIFT_SLIDER,
                                            dbc.Button("Run Correlation", 
                                                       id="corr_shift_button", 
                                                       className="btn btn-success")],
                                            align="center",
                                            style={"background-color": "#073642", "border-radius": "0.3rem"}),
                                
                                          dcc.Loading(
                                              dcc.Graph(id="corr_shift_matrix_plot")   
                                          )             
                            ])
                        )
             ]

CORR_CONCLUSIONS = html.Div(dcc.Markdown(conclusion_texts.correlation_conclusion), id="corr-conclusions")


# D CAUSALITY ANALYSIS

CAUS_SEASONAL_DROPDOWN = html.Div([
                        dcc.Dropdown(id='caus_seasonal_dropdown',
                                     options=column_small_labels,
                                     value="bitcoin_Price")
                     ,], style={"width":"20%"})



CAUS_SEASONAL_PLOT = [dbc.CardHeader(html.H5("Seasonal Decomposition")),
              dbc.CardBody(html.Div(children=[
                                        dbc.Row(children=[
                                                html.Label("Seasonal Decomposition for:",
                                                    style={"padding-left":20,
                                                           "padding": 10}), 
                                                CAUS_SEASONAL_DROPDOWN],
                                            align="center",
                                            style={"background-color": "#073642", "border-radius": "0.3rem"}),
                                            
                                            dcc.Loading(
                                                dcc.Graph(id="caus_seasonal_plot",
                                                          figure=SEASON_PLOT))
                                         ]))
                     ]

CAUS_GRANGER_PLOT = [dbc.CardHeader(html.H5("Granger Causality with Time Lag of 30 Days")),
                     dbc.CardBody(
                         html.Div(children=[
                                         dbc.Row( 
                                             html.Div(
                                                 dcc.Markdown(conclusion_texts.granger_prob_expl,
                                                              style={"padding-left":20, "padding": 10}), 
                                                 id="granger_expl"),
                                         
                                            align="center",
                                            style={"background-color": "#073642", "border-radius": "0.3rem"}),
                                        dcc.Loading(
                                            dcc.Graph(id="caus_granger_plot",
                                                      figure=GRANGER_PLOT))
                         ]
                                         ))]


CAUS_CONCLUSIONS = html.Div(dcc.Markdown(conclusion_texts.stationarity_causality_conclusion), id="caus-conclusions")


# E MODEL EVALUATION



MODEL_SARIMAX_EVAL = [dbc.CardHeader(html.H5("Model SARIMAX Cross Validation")),
                     dbc.CardBody(
                         html.Div(children=[dbc.Row( 
                                             html.Div(
                                                 dcc.Markdown("Used Features:\n\n  * {}".format("\n  * ".join(do_big.opt_ari_feat)),
                                                              style={"padding-left":20, "padding": 10}, 
                                                              id="sarimax_features")),
                                         
                                            align="center",
                                            style={"background-color": "#073642", "border-radius": "0.3rem"}),
                                        dcc.Loading(
                                            dcc.Graph(id="sarimax_cross_validation",
                                                      figure=CROSS_VAL_ARIMAX))
                         ]
                                         ))]

MODEL_GRU_EVAL = [dbc.CardHeader(html.H5("Model GRU Cross Validation")),
                     dbc.CardBody(
                         html.Div(children=[dbc.Row( 
                                             html.Div(
                                                 dcc.Markdown("Used Features:\n\n  * {}".format("\n  * ".join(do_big.opt_gru_feat)),
                                                              style={"padding-left":20, "padding": 10}, 
                                                              id="gru_features")),
                                         
                                            align="center",
                                            style={"background-color": "#073642", "border-radius": "0.3rem"}),
                                        dcc.Loading(
                                            dcc.Graph(id="gru_cross_validation",
                                                      figure=CROSS_VAL_GRU))
                         ]
                                         ))]


MODEL_CONCLUSIONS = html.Div(dcc.Markdown(conclusion_texts.model_eval_conclusion), id="model-conclusions")


# F FORECAST

FORE_DAYS_PICKER =  html.Div([   
    dcc.DatePickerSingle(
            id='fore_days_picker',
            min_date_allowed=FORE_DAYS[0],
            max_date_allowed=FORE_DAYS[-1],
            initial_visible_month=FORE_DAYS[0],
            date=FORE_DAYS[30]
        ),])


FORE_SENTIMENTS = [dbc.CardHeader(dbc.Row([html.P("TWITTER SENTIMENTS", id="sent_header")]),style={"height":"40px"}),
                   dbc.Spinner(color="info", children=[dbc.CardBody(html.P("EMPTY"), id="card_sents", style={"height":"130px",
                                                                         "background-color":"#073642"})])]

FORE_TRENDS = [dbc.CardHeader(dbc.Row([html.P("GOOGLE TRENDS", id="trend_header")]), style={"height":"40px"}),
                   dbc.Spinner(color="info", children=[dbc.CardBody(html.P("EMPTY"), id="card_trends", style={"height":"130px",
                                                                         "background-color":"#073642"})])]

FORE_STOCKS = [dbc.CardHeader(dbc.Row([html.P("STOCKS", id="stocks_header")]), style={"height":"40px"}),
                   dbc.Spinner(color="info", children=[dbc.CardBody(html.P("EMPTY"), id="card_stocks", style={"height":"130px",
                                                                         "background-color":"#073642"})])]

FORE_PAST_SLIDER =  [dbc.CardHeader(dbc.Row([html.P("Past Timespan:")]), style={"height":"40px"}),
                        dbc.CardBody(children=[
                            dbc.Col(
                                dcc.Slider(
                                    id='fore_past_slider',
                                    #updatemode = "drag",
                                    vertical = True,
                                    min=0,
                                    max=30,
                                    value=20,
                                    tooltip={"always_visible":False,
                                             "placement":"left"},
                                    verticalHeight=120),
                                 width={"size": 6, "offset": 3},
                            )
                    ],style={"height":"130px",
                             "background-color":"#073642"})]

FORE_ALL = [dbc.CardHeader(html.H5("Forecasting and Parameters for Day")),
                        dbc.CardBody( 
                            html.Div(children=[
                                        dbc.Row(children=[
                                            html.Label("Day to Predict:",
                                                style={"padding-left":20,
                                                       "padding": 10}), 
                                            FORE_DAYS_PICKER,
                                            html.Label("Charts:",
                                                style={"padding-left":20,
                                                       "padding": 10}),
                                            dbc.Col([
                                                dcc.Checklist(
                                                        id="boll_check",
                                                        options=[
                                                            {'label': 'Bollinger Bands', 
                                                             'value': 'boll'},], value=["boll"]),
                                            ]),
                                            dbc.Col([
                                                dcc.Checklist(
                                                        id="sarimax_check",
                                                        options=[
                                                            {'label': 'Sarimax Prediction', 
                                                             'value': 'ari'},], value=["ari"]),
                                                dcc.Checklist(
                                                        id="sarimax_ma",
                                                        options=[
                                                            {'label': 'Sarimax Moving Average', 
                                                             'value': 'ma'},], value=["ma"]),
                                            ]),
                                            dbc.Col([
                                                dcc.Checklist(
                                                    id="gru_check",
                                                    options=[
                                                        {'label': 'GRU Prediction', 
                                                         'value': 'gru'},], value=["gru"]),
                                            ]),
                                            ],
                                            align="center",
                                            style={"background-color": "#073642", "border-radius": "0.3rem"}),
                                
                                          dcc.Loading(
                                              dcc.Graph(id="fore_plot")
                                                       
                                          ),
                                          dbc.Row(children=[html.Div(FORE_PAST_SLIDER),
                                                            dbc.Col(FORE_SENTIMENTS),
                                                            dbc.Col(FORE_TRENDS),
                                                            dbc.Col(FORE_STOCKS)], style={"padding-top":"10px"})
                            ])
                        )
             ]



BUDGET_SLIDER = dcc.Slider(
                                    id='sim_budget',
                                    #updatemode = "drag",
                                    vertical = True,
                                    min=1000,
                                    max=1000000,
                                    value=100000,
                                    tooltip={"always_visible":True,
                                             "placement":"bottom"},
                                    verticalHeight=120)

MINMAX_DIST_SLIDER = dcc.Slider(
                                    id='maxmin_dist',
                                    #updatemode = "drag",
                                    vertical = True,
                                    min=1,
                                    max=10,
                                    value=5,
                                    tooltip={"always_visible":True,
                                             "placement":"bottom"},
                                    verticalHeight=120)

MINMAX_NEIGH_SLIDER = dcc.Slider(
                                    id='maxmin_neigh',
                                    #updatemode = "drag",
                                    vertical = True,
                                    min=1,
                                    max=5,
                                    value=1,
                                    tooltip={"always_visible":True,
                                             "placement":"bottom"},
                                    verticalHeight=120)

GRU_WINDOW_SLIDER = dcc.Slider(
                                    id='gru_window',
                                    #updatemode = "drag",
                                    vertical = True,
                                    min=3,
                                    max=20,
                                    value=12,
                                    tooltip={"always_visible":True,
                                             "placement":"bottom"},
                                    verticalHeight=120)



INPUT_BUDGET = [dbc.CardHeader(dbc.Row([html.P("START BUDGET")]),style={"height":"40px"}),
                   dbc.Spinner(color="info", children=[dbc.CardBody(html.Div(id="input_budget"), style={"height":"130px",
                                                                         "background-color":"#073642"})])]

MAX_BUDGET = [dbc.CardHeader(dbc.Row([html.P("MAX BUDGET")]),style={"height":"40px"}),
                   dbc.Spinner(color="info", children=[dbc.CardBody(html.Div(id="max_budget"), style={"height":"130px",
                                                                         "background-color":"#073642"})])]

MIN_BUDGET = [dbc.CardHeader(dbc.Row([html.P("MIN BUDGET")]),style={"height":"40px"}),
                   dbc.Spinner(color="info", children=[dbc.CardBody(html.Div(id="min_budget"), style={"height":"130px",
                                                                         "background-color":"#073642"})])]

PROFIT_BUDGET = [dbc.CardHeader(dbc.Row([html.P("PROFIT")]),style={"height":"40px"}),
                   dbc.Spinner(color="info", children=[dbc.CardBody(html.Div(id="profit_budget"), style={"height":"130px",
                                                                         "background-color":"#073642"})])]


BUY_SELL_SIM = [dbc.CardHeader(html.H5("Bitcoin Buy And Sell Simulation (GRU) between March 2019 - March 2020")),
                        dbc.CardBody( 
                            html.Div(children=[
                                        dbc.Row(children=[
                                            html.Label("Budget:",
                                                style={"padding-left":20,
                                                       "padding": 10}), 
                                            BUDGET_SLIDER,
                                            html.Label("Distance Max/Min:",
                                                style={"padding-left":20,
                                                       "padding": 10}),
                                            MINMAX_DIST_SLIDER,
                                            html.Label("Neighbors Max/Min:",
                                                style={"padding-left":20,
                                                       "padding": 10}),
                                            MINMAX_NEIGH_SLIDER,
                                            html.Label("Window Smooth:",
                                                style={"padding-left":20,
                                                       "padding": 10}),
                                            GRU_WINDOW_SLIDER,
                                          
                                            dbc.Button("Run Simulation", 
                                                       id="sim_button", 
                                                       style={"padding-left":20,
                                                              "padding":10},
                                                       className="btn btn-success"),
                                            
                                            ],
                                            align="center",
                                            style={"background-color": "#073642", "border-radius": "0.3rem", "width":"100%"}),
                                
                                        dbc.Row(children=[dbc.Col(INPUT_BUDGET),
                                                          dbc.Col(MAX_BUDGET),
                                                          dbc.Col(MIN_BUDGET),
                                                          dbc.Col(PROFIT_BUDGET)], style={"padding-top":"10px"}),
                                        dbc.Spinner(color="info", 
                                                      type="grow",
                                                      style={"width": "10rem", "height": "10rem"},
                                                      children=[
                                                                dcc.Graph(id="sim_plot")],
                                        ),
                                         
                            ])
                        )
             ]




CHANCES_ROADMAP =  html.Div(dcc.Markdown(conclusion_texts.chances_roadmap), id="model-conclusions")


### WEBAPP INIT ###

app = dash.Dash(__name__, 
                external_stylesheets=[dbc.themes.SOLAR], 
                url_base_pathname="/bitcoinprediction/",
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
acc_input = [Input(f"group-{i}-toggle", "n_clicks") for i in acc_str_list]
acc_input.extend([Input(f"slider-{i}", "value") for i in acc_str_list])
@app.callback(
    Output("right_column_loading", "children"),
    acc_input,
    [State("right_column_loading", "children")],
)    
def show_plot(acc_01, acc_02, acc_03, acc_04, acc_05, acc_06,
              sli_01, sli_02, sli_03, sli_04, sli_05, sli_06,
              figure):
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
        if sli_01 == 1:
            return INTRODUCTION
        if sli_01 == 0:
            return RESOURCES
        
    elif (acc_str_list[1] in element_id):
        if sli_02 == 1:
            return EXP_CHART_PLOT
        if sli_02 == 0:
            return VIEW_CONCLUSIONS
        
    elif (acc_str_list[2] in element_id):
        if sli_03 == 2:
            return CORR_01_CHART_PLOT
        elif sli_03 == 1:
            return CORR_02_CHART_PLOT
        elif sli_03 == 0:
            return CORR_CONCLUSIONS
        
    elif (acc_str_list[3] in element_id):
        if sli_04 == 2:
            return CAUS_SEASONAL_PLOT
        if sli_04 == 1:
            return CAUS_GRANGER_PLOT
        if sli_04 == 0:
            return CAUS_CONCLUSIONS
        
    elif (acc_str_list[4] in element_id):
        if sli_05 == 2:
            return MODEL_SARIMAX_EVAL
        if sli_05 == 1:
            return MODEL_GRU_EVAL
        if sli_05 == 0:
            return MODEL_CONCLUSIONS
        
    elif (acc_str_list[5] in element_id):
        if sli_06 == 2:
            return FORE_ALL
        if sli_06 == 1:
            return BUY_SELL_SIM
        if sli_06 == 0:
            return CHANCES_ROADMAP
    else:
        return FORE_ALL

        
# Control Accordion animation
@app.callback(
    [Output(f"collapse-{i}", "is_open") for i in acc_str_list],
    [Input(f"group-{i}-toggle", "n_clicks") for i in acc_str_list],
    [State(f"collapse-{i}", "is_open") for i in acc_str_list],
)
def toggle_accordion(n1, n2, n3, n4, n5, n6,
                     is_open1, is_open2, is_open3, is_open4, is_open5, is_open6):
    '''
    This function collapses the single Accordion Buttons based on click events
    '''
    ctx = dash.callback_context

    if not ctx.triggered:
        return ""
    else:
        button_id = ctx.triggered[0]["prop_id"].split(".")[0]

    if (acc_str_list[0] in button_id) and n1:
        return not is_open1, False, False, False, False, False
    elif (acc_str_list[1] in button_id) and n2:
        return False, not is_open2, False, False, False, False
    elif (acc_str_list[2] in button_id) and n3:
        return False, False, not is_open3, False, False, False
    elif (acc_str_list[3] in button_id) and n4:
        return False, False, False, not is_open4, False, False
    elif (acc_str_list[4] in button_id) and n5:
        return False, False, False, False, not is_open5, False
    elif (acc_str_list[5] in button_id) and n6:
        return False, False, False, False, False, not is_open6
    else:
        return False, False, False, False, False, not is_open6



# List for Output of Slider Subtitle HTML elements
output=[]
output.extend([Output(f"slidersub-{i}", "children") for i in acc_str_list])

# List for State of Slider Subtitle HTML elements
state=[]
state.extend([State(f"slidersub-{i}", "children") for i in acc_str_list])


# Control Subtitle Output below submenu slider
@app.callback(
    output,
    acc_input,
    state,
)    
def update_sub(acc_01, acc_02, acc_03, acc_04, acc_05, acc_06,
              sli_01, sli_02, sli_03, sli_04, sli_05, sli_06,
              slisub_01, slisub_02, slisub_03, slisub_04, slisub_05, slisub_06):
    '''
    Based on click events on accordion buttons and submenu sliders the subtitle
    texts below each submenu slider will be updated
    '''
    ctx = dash.callback_context

    if not ctx.triggered:
        return "", "", "", "", "", ""
    else:
        element_id = ctx.triggered[0]["prop_id"].split(".")[0]

        
    if (acc_str_list[0] in element_id):
        if sli_01 == 1:
            return "See Explanation for this Project", "", "", "", "", ""
        elif sli_01 == 0:
            return "See Used Resources", "", "", "", "", ""
    elif (acc_str_list[1] in element_id):
        if sli_02 == 1:
            return "", "See all investigated Input Time Series", "", "", "", ""
        elif sli_02 == 0:
            return "","See Conclusions about Input Time Series","", "", "", ""
    elif (acc_str_list[2] in element_id):
        if sli_03 == 2:
            return "","", "See correlational Analysis of Time Series", "", "", ""
        elif sli_03 == 1:
            return "","See correlational between timeshifted Time Series", "", "", "", ""
        elif sli_03 == 0:
            return "", "", "See Conclusions resulting from Correlation Analysis", "", "", ""
    elif (acc_str_list[3] in element_id):
        if sli_04 == 2:
            return "", "", "", "See Seasonal Decomposition for Time Series", "", ""
        elif sli_04 == 1:
            return "", "", "", "See Granger Causality for Time Series", "", ""
        elif sli_04 == 0:
            return "", "", "", "See Conclusions resulting from Causality Analysis", "", ""
    elif (acc_str_list[4] in element_id):
        if sli_05 == 2:
            return "", "", "", "", "See Cross Validation Results for SARIMAX Model", ""
        elif sli_05 == 1:
            return "", "", "", "", "See Cross Validation Results for GRU Model", ""
        elif sli_05 == 0:
            return "", "", "", "", "See Conclusions for Model Evaluation", ""
    elif (acc_str_list[5] in element_id):
        if sli_06 == 2:
            return "", "", "", "", "", "See manual daily forecast for Bitcoin Price"
        elif sli_06 == 1:
            return "", "", "", "", "", "See Buy & Sell Market Simulation"
        elif sli_06 == 0:
            return "", "", "", "", "", "See Conclusions and Outlook"
    else:
        return "", "", "", "", "", "See manual daily forecast for Bitcoin Price"

    
# controls green color of menu dot    
@app.callback(
    [Output(f"spandot-{i}", "style") for i in acc_str_list],
    [Input(f"group-{i}-toggle", "n_clicks") for i in acc_str_list],
    [State(f"spandot-{i}", "style") for i in acc_str_list],
)
def toggle_active_dot(n1, n2, n3, n4, n5, n6, 
                      active1, active2, active3, active4, active5, active6):
    '''
    Based on click events on the accordion button the style of the spandot in each accordion button
    will be updated
    '''
    
    sty_na={"height": "15px", 
           "width": "15px", 
           "background-color": "#bbb", 
           "border-radius": "50%",
            }
    
    sty_a={"height": "15px", 
           "width": "15px", 
           "background-color": "#00FF00", 
           "border-radius": "50%",
            }
    
    ctx = dash.callback_context

    if not ctx.triggered:
        return ""
    else:
        button_id = ctx.triggered[0]["prop_id"].split(".")[0]

    if (acc_str_list[0] in button_id) and n1:
        return sty_a, sty_na, sty_na, sty_na, sty_na, sty_na
    elif (acc_str_list[1] in button_id) and n2:
        return sty_na, sty_a, sty_na, sty_na, sty_na, sty_na
    elif (acc_str_list[2] in button_id) and n3:
        return sty_na, sty_na, sty_a, sty_na, sty_na, sty_na
    elif (acc_str_list[3] in button_id) and n4:
        return sty_na, sty_na, sty_na, sty_a, sty_na, sty_na 
    elif (acc_str_list[4] in button_id) and n5:
        return sty_na, sty_na, sty_na, sty_na, sty_a, sty_na 
    elif (acc_str_list[5] in button_id) and n6:
        return sty_na, sty_na, sty_na, sty_na, sty_na, sty_a
    else:
        return sty_na, sty_na, sty_na, sty_na, sty_na, sty_a
           

# belongs to F, outputs forecast
@app.callback(
    Output("fore_plot", "figure"),
    [Input("fore_days_picker", "date"),
     Input("boll_check", "value"),
     Input("sarimax_check", "value"),
     Input("sarimax_ma", "value"),
     Input("gru_check", "value")], 
    [State("fore_plot", "figure")])
def plot_forecast(curr_day, boll_check, arimax_check, sarimax_ma, gru_check, figure):
    '''
    Will return forecast plot and will be updated based on 
    date picker and several checkboxes
    '''
    if sarimax_ma:
        sarimax_ma_bool = True
    else:
        sarimax_ma_bool = False
          
    if boll_check:
        boll_bool = True
    else:
        boll_bool = False
        
    
    real_price, real_price_30 = do_big.get_real_price(curr_day, shift=-31)
    
    fig=""
    
    if arimax_check:
        ari_fore = do_big.ari_forecast(curr_day , shift=-31)
        
        fig = ph.get_ari_plot(df=ari_fore, fig=fig, conf_int=False, mean_averaged=sarimax_ma_bool)
        
    if gru_check:
        gru_fore = do_big.gru_forecast(curr_day, shift=-31)
        fig = ph.get_gru_plot(df=gru_fore, fig=fig)
    

    return ph.price_plot(real_price, 
                         real_30=real_price_30, 
                         fig=fig, 
                         boll=boll_bool,
                         height=500,
                         dash=True)

    
# belongs to F, controls growth indexes in forecast plot
@app.callback(
    [Output("card_sents", "children"),
     Output("sent_header", "children"),
     Output("card_trends", "children"),
     Output("trend_header", "children"),
     Output("card_stocks", "children"),
     Output("stocks_header", "children")],
    [Input("fore_days_picker", "date"),
     Input("fore_past_slider", "value")], 
    [State("fore_plot", "figure")])
def fill_fore_blocks(curr_day, past ,figure):   
    '''
    Returns content for growth indicators below forecast plot, will be updated by
    date picker and slider for adjusting lookback to past
    '''
    
    sentiments = ["economy_pos_sents", 
                  "bitcoin_pos_sents"] 
     
    trends = ["bitcoin_Google_Trends", 
              "cryptocurrency_Google_Trends"] 
     
    stocks = ["bitcoin_Price",
              "sp500_Price", 
              "alibaba_Price", 
              "amazon_Price"]
    
    all_indicators = []
    all_indicators.extend(sentiments)
    all_indicators.extend(trends)
    all_indicators.extend(stocks)
    
    past = past*-1
    growth_dict = do_big.get_growth(curr_day, past, all_indicators)
    
    card_sents = build_growth_content(growth_dict, sentiments)
    sents_header = "TWITTER SENTIMENTS last {} days".format(abs(past))
    card_trends = build_growth_content(growth_dict, trends)
    trends_header = "GOOGLE TRENDS last {} days".format(abs(past))
    card_stocks = build_growth_content(growth_dict, stocks)
    stocks_header = "STOCKS last {} days".format(abs(past))

    return card_sents, sents_header, card_trends, trends_header, card_stocks, stocks_header

                                     
                                     
# belongs to F, outputs forecast
@app.callback(
    [Output("sim_plot", "figure"),
     Output("input_budget", "children"),
     Output("max_budget", "children"),
     Output("min_budget", "children"),
     Output("profit_budget", "children")],
    [Input("sim_button", "n_clicks")], 
    [State("sim_budget", "value"),
     State("maxmin_dist", "value"),
     State("maxmin_neigh", "value"),
     State("gru_window", "value"),
     State("sim_plot", "figure")])
def plot_simulation(n_clicks, sim_budget, min_max_dist, num_neigh, gru_window, figure):
    '''
    Will return plot for simulation with multiple slider values for adjusting simulation.
    '''
    
    result_df = do_big.simulate_buy_sell(sim_budget, min_max_dist, num_neigh, gru_window ,future_offset_val=31)
    
    in_budget = html.P("{}".format('${:,.2f}'.format(sim_budget)), style={"font-weight":"bold", "font-size":"25px"})
                       
    max_val = result_df.budget.max()
    min_val = pd.Series(result_df.budget.unique()).nsmallest(2).iloc[-1]
    out_val = result_df.budget[-1]
    
    output_budgets = []
    for val in [max_val, min_val, out_val]:
        if val >= sim_budget:
            color="green"
        else:
            color = "red"
        cur = '${:,.2f}'.format(val)
        output_budgets.append(html.P("{}".format(cur),style={"color":color,"font-weight":"bold", "font-size":"25px"}))
    
                       
    return ph.plot_buy_sell_sim(result_df,dash=True), in_budget, output_budgets[0], output_budgets[1], output_budgets[2]                                     

                                     
# belongs to D, plots seasonal decomposition plot
@app.callback(
    Output("caus_seasonal_plot", "figure"),
    [Input("caus_seasonal_dropdown", "value")],
    [State("caus_seasonal_plot", "figure")]
)
def ret_caus_seasonal_plot(dropdown, figure):
    '''
    Returns Seasonal decomposition plot and will be updated
    by triggering dropdown choosing a single time series
    '''
    return ph.return_season_plot(do_big.chart_df[small_col_set], dropdown, title="", dash=True)
    

# belongs to C, plot manual shifted correlation plot
@app.callback(
    Output("corr_shift_matrix_plot", "figure"),
    [Input("corr_shift_button", "n_clicks")],
    [State("corr_shift_dropdown", "value"), State("corr_shift_slider", "value")]
)
def ret_corr_shift_plot(n, dropdown, slider):
    '''
    Returns correlation heatmap with shifted timeseries and will be updated by 
    triggering dropdown to choose fixed timeseries and slider to choose shift of other timeseries
    '''
    
    small_set_copy = small_col_set.copy()
    do_big.fixed_cols = dropdown
    shift_df = do_big.single_shift(slider, small_set_copy)
    corr = shift_df.corr()
    
    return ph.return_shift_corr(corr, fixed=dropdown, output="single", dash=True)
    
    

    
if __name__ == "__main__":
    app.run_server(debug=True,  port=8050, host="0.0.0.0")
