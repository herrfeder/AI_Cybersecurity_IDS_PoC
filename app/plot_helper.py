import plotly.graph_objects as go
from plotly.subplots import make_subplots
from statsmodels.tsa.seasonal import seasonal_decompose
from pandas.tseries.offsets import DateOffset
import pandas as pd
import numpy as np


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



def get_gru_plot(df, fig="", title="", dash=False):
    '''
    Get Plot for GRU prediction Chart.
    
    INPUT:
        df - (DataFrame) with Datetime index and single Timeseries column
        fig - (plotly Figure) On passing a fig the plot will be added to an existing plot in this argument
        title - (str) Figure Title
        dash - (bool) On True, will return Figure with Web App specific layout
               On False, will return plain figure
    OUTPUT:
        fig - (plotly Figure) plot object that needs to be passed to dash figure attribute
    '''
    
    if not fig:
        fig = make_subplots(
                        rows=1, 
                        cols=1, 
                        shared_xaxes=True)
        
        
    fig.add_trace(go.Scatter(x=df.index, 
                             y=df[0],
                             line=dict(color='blue'),
                             name="GRU Prediction"), row=1, col=1)
    
    if dash:
        return apply_layout(fig, title)
    else:
        return fig
    


def get_ari_plot(df, fig="", title="", offset=31, conf_int=False, mean_averaged=True, dash=False):
    '''
    Get Plot for SARIMAX prediction Chart with optional Confidence Intervals.
    The offset is needed to align the prediction with the true plot.
    
    INPUT:
        df - (statsmodels SARIMAX object) in df.predicted_mean lies Datetime index and single Timeseries column
             in df.conf_int() are the confidence intervals for prediction
        fig - (plotly Figure) On passing a fig the plot will be added to an existing plot in this argument
        title - (str) Figure Title
        offset - (int) due to some shift during forecast we have to shift the prediction for beeing aligned
                 with the true plot
        conf_int - (bool) on True, adding Confidence Intervals for prediction to plot
                   on False, not
        dash - (bool) On True, will return Figure with Web App specific layout
               On False, will return plain figure
    OUTPUT:
        fig - (plotly Figure) plot object that needs to be passed to dash figure attribute
    '''
    
    if not fig:
        fig = make_subplots(
                        rows=1, 
                        cols=1, 
                        shared_xaxes=True)
        
    if mean_averaged:
        df_mean = df.predicted_mean.rolling(10, min_periods=1).mean()
    else:
        df_mean = df.predicted_mean
        
    fig.add_trace(go.Scatter(x=df.predicted_mean.index + DateOffset(offset), 
                             y=df_mean,
                             line=dict(color='green'),
                             name="SARIMAX Prediction"), row=1, col=1)
    
    if conf_int:
        
        fig.add_trace(go.Scatter(x=df.predicted_mean.index + DateOffset(offset), 
                             y=df.conf_int()["lower bitcoin_Price"],
                             fill='tonexty',
                             fillcolor='rgba(166, 217, 193,0.2)',
                             line=dict(color='rgba(255,255,255,0)'),
                             name="SARIMAX Lower Confidence Interval"))


        fig.add_trace(go.Scatter(x=df.predicted_mean.index + DateOffset(offset), 
                             y=df.conf_int()["upper bitcoin_Price"],
                             fill='tonexty',
                             fillcolor='rgba(166, 217, 193,0.2)',
                             line=dict(color='rgba(255,255,255,0)'),
                             name="SARIMAX Higher Confidence Interval"))
        
    
    if dash:
        return apply_layout(fig, title)
    else:
        return fig
    

def price_plot(df, real_30=pd.DataFrame(),fig="", title="", boll=True, dash=False, height=800,names=["BTC Price",
                                           "BTC 30 Day Moving Average",
                                           "BTC Upper Bollinger Band",
                                           "BTC Lower Bollinger Band"]):
    '''
    Plotting the real price, optionally the offset into future for comparison to prediction, 
    optionally the Bollinger Bands.
    
    INPUT:
        df - (DataFrame) with Datetime index and single Timeseries column
        real_30 - (DataFrame) with Datetime index and single Timeseries column
        fig - (plotly Figure) On passing a fig the plot will be added to an existing plot in this argument
        title - (str) Figure Title
        boll - (bool) On True, plot Bollinger Bands
               On False, Not
        dash - (bool) On True, will return Figure with Web App specific layout
               On False, will return plain figure
        names - (list of str) Legend names for plots
    OUTPUT:
        fig - (plotly Figure) plot object that needs to be passed to dash figure attribute

    '''
    if not fig:
        fig = make_subplots(
                        rows=1, 
                        cols=1, 
                        shared_xaxes=True)
     
    fig.add_trace(go.Scatter(x=df.index, 
                             y=df['bitcoin_Price'],
                             line=dict(color='#2cfec1'),
                             name=names[0]), row=1, col=1)
    
    if not real_30.empty:
        fig.add_trace(go.Scatter(x=real_30.index,
                                 y=real_30['bitcoin_Price'],
                                 name="Future Real Bitcoin Price",
                                 line=dict(color='grey', dash='dot')), row=1, col=1)
    if boll:
        fig.add_trace(go.Scatter(x=df.index, 
                                 y=df['bitcoin_30_day_ma'],
                                 name=names[1]), row=1, col=1)

        fig.add_trace(go.Scatter(x=df.index, 
                                 y=df['bitcoin_boll_upp'],
                                 fill='tonexty',
                                 fillcolor='rgba(231,107,243,0.2)',
                                 line=dict(color='rgba(255,255,255,0)'),
                                 name=names[2]), row=1, col=1)

        fig.add_trace(go.Scatter(x=df.index, 
                                 y=df['bitcoin_boll_low'],
                                 fill='tonexty',
                                 fillcolor='rgba(231,50,243,0.2)',
                                 line=dict(color='rgba(255,255,255,0)'),
                                 name=names[3]), row=1, col=1) 
        
        
    if dash:
        return apply_layout(fig, title, height)
    else:
        return fig


def exploratory_plot(df, title="",dash=False):
    '''
    Plotting the charts for all input time series. The used columns for plotting
    are hardcoded and could be updated to a more generic way.
    
    INPUT:
        df - (DataFrame) with Datetime index and multiple Timeseries columns
        title - (str) Figure Title
        dash - (bool) On True, will return Figure with Web App specific layout
               On False, will return plain figure
    OUTPUT:
        fig - (plotly Figure) plot object that needs to be passed to dash figure attribute

    '''
    
    
    fig = make_subplots(
                        rows=4, 
                        cols=1, 
                        shared_xaxes=True, 
                        vertical_spacing=0.08,
                        subplot_titles=("Bitcoin Historic Price Chart with Bollinger Bands (30-day)", 
                                        "Other Normalized Stock Price Historic Charts", 
                                        "Historic Google Trends", 
                                        "Historic Sentiments Twitter")
                        )
    
    fig = price_plot(df, fig=fig)

    y_2_list = ["sp500_Price_norm", 
                "dax_Price_norm",
                "googl_Price_norm",
                "gold_Price_norm",
                "amazon_Price_norm",
                "alibaba_Price_norm"]
    
    name_2_list = ["SP500 Normed Close",
                   "DAX Normed Close",
                   "GOOGLE Normed Close",
                   "GOLD Normed Close",
                   "AMAZON Normed Close",
                   "ALIBABA Normed Close"]
    
    for y, name in zip(y_2_list, name_2_list):
        fig.add_trace(go.Scatter(x=df.index, 
                                y=df[y],
                                name=name), row=2, col=1)

   
    fig.add_trace(go.Scatter(x=df.index,
                             y=df["bitcoin_Google_Trends"],
                             name="'Bitcoin' Google Trends"), row=3, col=1)

    fig.add_trace(go.Scatter(x=df.index,
                             y=df["trading_Google_Trends"],
                             name="'Trading' Google Trends"), row=3, col=1)

    fig.add_trace(go.Scatter(x=df.index,
                             y=df["cryptocurrency_Google_Trends"],
                             name="'Cryptocurrency' Google Trends"), row=3, col=1)

    fig.add_trace(go.Scatter(x=df.index,
                             y=df["bitcoin_quot_sents"],
                             name="'Bitcoin' Sentiments"), row=4, col=1)

    fig.add_trace(go.Scatter(x=df.index,
                             y=df["economy_quot_sents"],
                             name="'#economy' Sentiments"), row=4, col=1)
    
    fig.update_yaxes(title_text="Absolute Price in $", row=1, col=1)
    fig.update_yaxes(title_text="Normalized Price", row=2, col=1)
    fig.update_yaxes(title_text="Number of Query per day", row=3, col=1)
    fig.update_yaxes(title_text="Normalized Sentiment Quotient Value", row=4, col=1)

    if dash:
        return apply_layout(fig, title)
    else:
        fig.show()

        
def return_season_plot(df, column, title="", dash=False):
    '''
    Plotting the charts for all input time series. The used columns for plotting
    are hardcoded and could be updated to a more generic way.
    
    INPUT:
        df - (DataFrame) with Datetime index and multiple Timeseries columns
        column - (str) column in input df to apply seasonal decomposition for
        title - (str) Figure Title
        dash - (bool) On True, will return Figure with Web App specific layout
               On False, will return plain figure
    OUTPUT:
        fig - (plotly Figure) returns object that needs to be passed to dash figure attribute
    '''
    
    series = pd.DataFrame(data=df[column].values, 
                          index=df.index, 
                          columns =[column]).dropna()

    result = seasonal_decompose(series.values, model='multiplicative', period=30)
    
    fig = make_subplots(
                        rows=3, 
                        cols=1, 
                        shared_xaxes=True, 
                        vertical_spacing=0.08,
                        )
    
    fig.add_trace(go.Scatter(x=df.index, 
                         y=result.trend,
                         name="Trend"), row=1, col=1)

    fig.add_trace(go.Scatter(x=df.index, 
                         y=result.resid,
                         name="Residuals"), row=2, col=1)

    fig.add_trace(go.Scatter(x=df.index, 
                             y=result.seasonal,
                             name="Seasonality"), row=3, col=1)
    
    fig.update_yaxes(title_text="Trend Of Chart", row=1, col=1)
    fig.update_yaxes(title_text="Residuals Of Chart", row=2, col=1)
    fig.update_yaxes(title_text="Seasonality Of Chart", row=3, col=1)
    
    
    if dash:
        return apply_layout(fig, title, height=800)
    else:
        fig.show()
        

def plot_val_heatmap(df, title="", height=1000, colormap="viridis", colorbar="Corr Coefficient",dash=False):
    '''
    Plots an Plotly based heatmap, based on the input of a DataFrame with Correlation Matrix content.
    
    INPUT:
        df - (DataFrame) that consists of Correlation Matrix
        title - (str) Figure Title
        height - (int) Figure Height
        colormap - (str) Colormap to use for heatmap
        colorbar - (str) Label for the colorbar
        dash - (bool) On True, will return Figure with Web App specific layout
               On False, will return plain figure
    OUTPUT:
        figure - (plotly Figure) returns object that needs to be passed to dash figure attribute
    '''
    
    coordinates = df.values.tolist()
    columns = list(df.columns)
    index = list(df.index)

    trace1 = {
            "type": "heatmap", 
            "x": columns, 
            "y": index, 
            "z": coordinates,
            "colorscale": colormap,
            "colorbar": {"title": colorbar}
            }

    data = trace1
    layout = {"title": title,
              "height": 1000}
    fig = go.Figure(dict(data=data, layout=layout))
    
    if dash:
        return apply_layout(fig, title, height)
    else:
        fig.show()
        
        
def return_shift_corr(corr, fixed="bitcoin_Price", output="multi", dash=False):
    '''
    
    INPUT:
        corr - (DataFrame) DataFrame with Correlation Matrix Content
        fixed - (str) Fixed column that we wan't to see the correlations for
        output - (str) on "multi" complete Correlation Heatmap will be returned
                 on "single" only row for fixed Column will be returned
        dash - (bool) On True, will return Figure with Web App specific layout
               On False, will return plain figure
    OUTPUT:
        figure - (plotly Figure) will run the Shifted Correlation through another function for applying to heatmap
    '''
    
    
    if output=="single":
        corr = corr[corr.index == fixed]
        return plot_val_heatmap(corr, height=400, dash=True)
    else:
        return plot_val_heatmap(corr, height=800,dash=True)
    

def return_granger_plot(df_path, title="", height=1000, colormap="viridis",dash=False):
    '''
    PreparesGranger Causality from static file (because of long algorithm runtime) for Heatmap Plot. 
    Will adjust input DataFrame for better visibility,
    as everything that is far above 0.06 will be NaN for having more detailed colorscale
    in the interesting range.
    
    
    INPUT:
        df_path - (str) relative path to granger plot csv
        title - (str) Figure Title
        height - (int) Figure Height
        colormap - (str) Colormap to use for heatmap
        dash - (bool) On True, will return Figure with Web App specific layout
               On False, will return plain figure
    OUTPUT:
        figure - (plotly Figure) will run the Granger data through another function for applying to heatmap
    '''
    
    
    granger_df = pd.read_csv(df_path)
    granger_df.set_index("Unnamed: 0", inplace=True)
    granger_df[granger_df > 0.06] = np.nan
    return plot_val_heatmap(granger_df, 
                            title=title, 
                            height=height, 
                            colormap=colormap,
                            colorbar="P-Value",
                            dash=dash)

def return_cross_val_plot(split_dict, title="", height=1200, dash=False):
    '''
    Will show the true plot and predicted plot for each cross validation split.
    
    
    INPUT:
         split_dict - (dict) Holds error values and time series arrays for each split
         title - (str) Figure Title
         height - (int) Figure Height
         dash - (bool) On True, will return Figure with Web App specific layout
               On False, will return plain figure
    OUTPUT:
         figure - (plotly Figure) will return figure with an subplot for every split
    '''
    params = ["CORR", "MSE", "RMSE", "R2"]
       
    valid_list = []
    forecast_list = []
    title_list = []
    for i in range(0,3):
        title_text = "Split {}<br>| ".format(i+1)
        for para in params:
            dict_val = split_dict.get("S_{}_{}".format(i,para),"") 
            if dict_val:
                title_text += para+" = "+str(np.round(dict_val,2))+" | "
        
        title_list.append(title_text)
        
        valid_plot = split_dict["S_{}_VALID".format(i)]
        
        if hasattr(valid_plot, "index"):
            valid_plot_index = valid_plot.index
        else:
            valid_plot_index = list(range(0, len(valid_plot)))
        
        valid_list.append(go.Scatter(x=valid_plot_index, 
                         y=valid_plot,
                         name="Real Bitcoin Price Split {}".format(i+1)))
        
        fore_plot = split_dict["S_{}_FORE".format(i)]
        
        if hasattr(fore_plot, "index"):
            fore_plot_index = valid_plot.index
        else:
            fore_plot_index = list(range(0, len(fore_plot)))
        
        forecast_list.append(go.Scatter(x=valid_plot_index, 
                         y=fore_plot,
                         name="Predicted Bitcoin Price Split {}".format(i+1)))
        
    fig = make_subplots(
                    rows=3, 
                    cols=1, 
                    vertical_spacing=0.08,
                    subplot_titles=(title_list[0], 
                                    title_list[1], 
                                    title_list[2]))
    
    index = 1
    for valid, forecast in zip(valid_list, forecast_list):
        fig.add_trace(valid, row=index, col=1)
        fig.add_trace(forecast, row=index, col=1)
        index = index + 1
    
  
    if dash:
        return apply_layout(fig, title, height=height)
    else:
        fig.show()
    
    
def plot_buy_sell_sim(df, title="", height=500,dash=False):
    '''
    Plots the results for the buy and sell simulation.
    This includes GRU growth prediction, real price plot and Buy and Sell triggers.
    
    INPUT:
        df - (DataFrame) with result columns from simulation
        dash - (bool) On True, will return Figure with Web App specific layout
               On False, will return plain figure
    OUTPUT:
        fig - (plotly Figure) with traces as described above
    '''
    
    
    fig = make_subplots(
                        rows=1, 
                        cols=1, 
                        shared_xaxes=True)
        
        
    fig.add_trace(go.Scatter(x=df.index, 
                                y=df["gru_growth"],
                                line=dict(color='#2cfec1'),
                                name="GRU Growth"), row=1, col=1)

    real_normed = (((df["curr_price"]*10)/1000)-40)

    fig.add_trace(go.Scatter(x=df.index, 
                                y=real_normed,
                                line=dict(color='grey'),
                                name="Real Price Scaled"), row=1, col=1)

    fig.add_trace(go.Scatter(x=df.index, 
                                y=df["buy_trigger"],
                                text = ["BTC: {}".format(np.round(x)) for x in df["bitcoin"]],
                                textposition = "bottom center",
                                marker=dict(color="crimson", size=10),
                                mode="markers+text",
                                name="Buy Trigger"), row=1, col=1)

    fig.add_trace(go.Scatter(x=df.index, 
                                y=df["sell_trigger"],
                                text = ["USD: {} $".format(np.round(x)) for x in df["budget"]],
                                textposition = "top center",
                                textfont=dict(color=["green" if x==1 else "red" for x in df["profit"]]),
                                marker=dict(color="yellow", size=10),
                                mode="markers+text",
                                name="Sell Trigger"), row=1, col=1)
    
    
    
    if dash:
        return apply_layout(fig, title, height=height)
    else:
        fig.show()