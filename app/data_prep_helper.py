import pandas as pd
import pathlib
import os
import sys
import numpy as np
import pickle
from math import sqrt
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model
from pandas.tseries.offsets import DateOffset
from datetime import datetime
from scipy.signal import argrelextrema



class ChartData():
    '''
    ChartData acts as the Data Preprocessing Unit. It reads all collected Time Series
            - Multiple Financial Chart Data from Investing.com with similiar structure
            - Self generated google trends dataset for several keywords
            - Self generated twitter sentiments datasets for keywords "bitcoin" and "#economy"
    and aggregates them into a single DataFrame chart_df with Datetime index.
    '''
    def __init__(self, window_size=30, chart_col="Price"):
        '''
        Reads input arguments and does the most of data input processing
            - Read CSV's into dict of dataframes df_d
            - Clean and prepare stock market datasets
            - Clean and prepare google trend dataset
            - Clean and prepare twitter sentiment datasets
            - Merging all datasets into single DataFrame (merge_dict_to_chart_df)
        
        INPUT:
            window_size - (int) Defines the Moving Average Window for generating the Bollinger Bands
            chart_col - (str/list of str) Gives the column/s that will be taken from the chart df input data
                        into the resulting chart_df
        '''
    
        charts = ["bitcoin_hist", "sp500_hist", "dax_hist", "googl_hist", "gold_hist", "alibaba_hist", "amazon_hist"]
        sents = ["bitcoin_sent_df", "economy_sent_df"]
        
        self.window_size = window_size
        
        self._chart_df = ""
        
        self.base_path = pathlib.Path(__file__).parent.resolve()
        self.data_path = os.path.join(self.base_path, "data")
        
        self.df_d = self.read_data_sources()
        for chart in charts:
            self.df_d[chart] = self.prep_charts(chart, norm=True)   
        
        self.df_d["trend_df"] = self.prepare_trend("trend_df")

        for sent in sents:
            self.df_d[sent] = self.prepare_sent(sent)

        self.merge_dict_to_chart_df(chart_col)
        
        
    @property
    def chart_df(self):
        return self._chart_df
    
    @chart_df.setter
    def chart_df(self, df):
        self._chart_df = df
        
    @property
    def df_d(self):
        return self._df_d
    
    @df_d.setter
    def df_d(self, df_dict):
        self._df_d = df_dict
        
        
    def read_data_sources(self):
        data_source_d = {}
        data_source_d["bitcoin_hist"] = pd.read_csv(
            os.path.join(self.data_path, "Bitcoin Historical Data - Investing.com.csv"))
        data_source_d["sp500_hist"] = pd.read_csv(
            os.path.join(self.data_path, "S&P 500 Historical Data.csv"))
        data_source_d["dax_hist"] = dax_hist = pd.read_csv(
            os.path.join(self.data_path, "DAX Historical Data.csv"))
        data_source_d["googl_hist"] = pd.read_csv(
            os.path.join(self.data_path,"GOOGL Historical Data.csv"))
        data_source_d["trend_df"] = pd.read_csv(
            os.path.join(self.data_path, "trends_bitcoin_cc_eth_trading_etf.csv"))
        data_source_d["bitcoin_sent_df"] = pd.read_csv(
            os.path.join(self.data_path, "bitcoin_sentiments.csv"))
        data_source_d["economy_sent_df"] = pd.read_csv(
            os.path.join(self.data_path, "economy_sentiments.csv"))
        data_source_d["gold_hist"] = pd.read_csv(
            os.path.join(self.data_path, "GOLD Historical Data.csv"))
        data_source_d["alibaba_hist"] = pd.read_csv(
            os.path.join(self.data_path, "BABA Historical Data.csv"))
        data_source_d["amazon_hist"] = pd.read_csv(
            os.path.join(self.data_path, "AMZN Historical Data.csv"))
        return data_source_d
    

    def apply_boll_bands(self, 
                         df_string="", 
                         price_col="Price", 
                         window_size=0, 
                         append_chart=False, 
                         ext_df=pd.DataFrame()):
        '''
        Calculates Bollinger Bands for chart_df or given external DataFrame
        
        INPUT:
            df_string - (str) key name for the DataFrame in the dataframe dict df_d to process Bollinger Bands for
            price_col - (str) Column name in the stock market DataFrame to calculate Bollinger Bands for
            window_size - (int) Override class window size for calculating Bollinger Bands
            append_chart - (bool) On True resulting columns will be attached and stored to class owned chart_df
                           on False resulting columns will be returned with class owned chart_df
            ext_df - (DataFrame) Will use external DataFrame for applying Bollinger Bands
        OUTPUT:
            dataframe - (DataFrame) On "append_chart=False" returns input DataFrame with concat columns
                        "prefix_30_day_ma", "prefix_30_day_std", "prefix_boll_up", "prefix_boll_low"
        '''
        try:
            if not ext_df.empty:
                df = ext_df
            else:
                df = self.df_d[df_string]
        except:
            print("Not found this DataFrame name")
            
        if window_size== 0:
            window_size = self.window_size
        
        prefix = df_string.split("_")[0]
        
        df["30_day_ma"] = df[price_col].rolling(window_size, min_periods=1).mean()
        df["30_day_std"] = df[price_col].rolling(window_size, min_periods=1).std()
        df["boll_upp"] = df['30_day_ma'] + (df['30_day_std'] * 2)
        df["boll_low"] = df['30_day_ma'] - (df['30_day_std'] * 2)
        
        if append_chart:
            self.append_to_chart_df(df[["30_day_ma", "30_day_std", "boll_upp", "boll_low"]], prefix)
        else:
            if not ext_df.empty:
                df.columns = ["{}_{}".format(prefix, col) if not col.startswith(prefix) else col for col in df.columns]
                return df
            else:
                return self.append_to_chart_df(df[["30_day_ma", "30_day_std", "boll_upp", "boll_low"]], prefix, inplace=False)
        
        
    def prep_charts(self, df_string, norm=False):
        '''
        Cleans and processes input stock chart datasets from Investing.com
            - by converting stringified numbers to float
            - converting human readable strings like "10k", "4M" to float
            - setting Datetime as index
        
        INPUT:
            df_string - (str) key name for the DataFrame in the dataframe dict df_d
            norm - (bool) On true adding an maximum normalized Price column
        OUTPUT:
            df - (DataFrame) returns input DataFrame with processed data
        '''
        try:
            df = self.df_d[df_string]
        except:
            print("Not found this DataFrame name")
        
        df["Price"] = df.apply(convert_values, args=("Price",), axis=1)
        df["Open"] = df.apply(convert_values, args=("Open",), axis=1)
        df["High"] = df.apply(convert_values, args=("High",), axis=1)
        df["Low"] = df.apply(convert_values, args=("Low",), axis=1)
        df["Vol."] = df.apply(convert_vol, args=("Vol.",), axis=1)


        df["Date"] = pd.to_datetime(df["Date"])
        df = df.sort_values(by="Date").reset_index()
        del df["index"]
        df = df.set_index("Date")

        if norm:
            df["Price_norm"] = df["Price"] / df["Price"].max()

        return df

    
    def prepare_trend(self, df_string):
        '''
        The Google Trends data consists of hourly collected data points.
        Setting Datetime index and resample hourly data to daily time series.
        
        
        INPUT:
            df_string - (str) key name for the DataFrame in the dataframe dict df_d
        OUTPUT:
            df - (DataFrame) returns input DataFrame with processed data
        '''
        try:
            df = self.df_d[df_string]
        except:
            print("Not found this DataFrame name")
            return
        
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index("date")
        df.index.name = "Date"
        df = df.resample("D").sum()
        
        return df
        
    def prepare_sent(self, df_string):
        '''
        Setting Datetime index and adding additional column with quotient of
        positive and negative sentiments.
        
        
        INPUT:
            df_string - (str) key name for the DataFrame in the dataframe dict df_d
        OUTPUT:
            df - (DataFrame) returns input DataFrame with processed data
        '''
        try:
            df = self.df_d[df_string]
        except:
            print("Not found this DataFrame name")
            return
        
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date")
        df.index.name = "Date"
        df["quot"] = df["pos"] / df["neg"]
        
        return df
    

    def merge_dict_to_chart_df(self, chart_col=["Price"]):
        '''
        Merges all input DataFrames in df_d into a single DataFrame chart_df
        with a joint Datetime daily index. All original column names get a prefix. 
        The selected DataFrame names are hardcoded right now and could be updated to a generic function.
        
        INPUT:
            chart_col - (list of str) Contains the column names that should be selected from the stock market
                        datasets
        OUTPUT:
            none - Result will be stored into class owned chart_df
        '''
        
        if not isinstance(chart_col, list):
            chart_col = [chart_col]
            
        self.chart_df = self.df_d["bitcoin_hist"][chart_col]
        self.chart_df.columns = ["bitcoin_{}".format(x) for x in self.chart_df.columns]

        for stock in ["sp500_hist", "dax_hist", "googl_hist", "gold_hist", "alibaba_hist", "amazon_hist"]:
            stock_name = stock.split("_")[0]
            self.chart_df = self.chart_df.merge(self.df_d[stock][chart_col], 
                                                left_index=True, 
                                                right_index=True)
            self.chart_df = self.chart_df.rename(columns={col:"{}_{}".format(stock_name, col) for col in chart_col})

        self.chart_df = self.chart_df.merge(self.df_d["trend_df"], 
                                            left_index=True, 
                                            right_index=True).drop(columns=["etf", "ethereum", "isPartial"])
        self.chart_df = self.chart_df.rename(columns={"bitcoin":"bitcoin_Google_Trends", 
                                                      "cryptocurrency":"cryptocurrency_Google_Trends",
                                                      "trading":"trading_Google_Trends"})

        for sent in ["bitcoin_sent_df", "economy_sent_df"]:
            sent_name = sent.split("_")[0]
            self.chart_df = self.chart_df.merge(self.df_d[sent], 
                                                left_index=True, 
                                                right_index=True).drop(columns=["length"])
            self.chart_df = self.chart_df.rename(columns={"pos": sent_name+"_pos_sents",
                                              "neg": sent_name+"_neg_sents",
                                              "quot": sent_name+"_quot_sents"})
            
            
        self.chart_df = self.chart_df.resample('D').interpolate()

            
    def append_to_chart_df(self, append_df, prefix_name, inplace=False):
        '''
        Will append an DataFrame with prefixed column names to chart_df and will store it there
        or return it to function caller. The concatination/merging will happen with
        Datetime index as key.
        
        INPUT:
            append_df - (DataFrame) DataFrame to append to chart_df, must have Datetime index
            prefix_name - (str) string to prepend to the column_names of append_df on merging
            inplace - (bool) on true appends to chart_df, on false returns to function caller
        OUTPUT:
            dataframe - (DataFrame) on inplace==False the merged DataFrame will be returned
            
        '''
        append_df.columns = ["{}_{}".format(prefix_name, col) for col in append_df.columns]
        if not inplace:
            return self.chart_df.merge(append_df, left_index=True, right_index=True)
        else:
            self.chart_df = self.chart_df.merge(append_df, left_index=True, right_index=True)
            
            
    def get_growth(self, day, past, cols=["bitcoin_Price"]):
        '''
        Calculates linear growth of timeseries values in timespan given by "past" from date
        given by "day". It will apply a moving average to the extracted span and
        returns the resulting growth percentage between first and last value of span.
        
        INPUT:
            day - (str "2019-01-01") date from which a timespan into past will be taken
            past - (int) timespan for growth calculation
            cols - (list of str) column/s for which the growth has to be calculated
        OUTPUT:
            growth_dict (dict) holds column names as keys and calculated growth as values
        '''
        
        
        past_days = past
        window = int(np.round(abs(past_days)/2))
        
        past_days_df = self.chart_df[self.chart_df.index < day].iloc[past_days:,:].rolling(window=window, min_periods=1 ).mean()
        
        if not isinstance(cols,list):
                cols = [cols]
        
        growth_dict = {}
        for col in cols:
            growth_dict[col] = np.round(100 - ((past_days_df[col][0]/past_days_df[col][-1])*100),2)
            
        return growth_dict
        
        

class ShiftChartData(ChartData):
    '''
    Takes over class methods from ChartData and adds functions for splitting and shifting data
    for Feature Engineering and Model preparation.
    '''
    def __init__(self, fixed_cols="bitcoin_Price", window_size=30, chart_col="Price"):
        '''
        Takes arguments from ChartData and adds fixed_cols to know which columns should be
        fixed while the others are shifted in time.
        
        INPUT:
            fixed_cols - (str/list of str) will be checked on existence in chart_df on any change
        '''
        super().__init__(window_size, chart_col)    
        
        self._fixed_cols = fixed_cols
    
    
    @property
    def fixed_cols(self):
        return self._fixed_cols
    
    
    @fixed_cols.setter
    def fixed_cols(self, fixed_cols):
        '''
        fixed_cols setter will check existence of all added columns
        '''
        if not isinstance(fixed_cols, list):
            fixed_cols = [fixed_cols]
            
        self._fixed_cols = self.check_fixed_cols(fixed_cols)
    
    
    def check_fixed_cols(self, fixed_cols):
        '''
        Checks on existence of given fixed_cols in chart_df. If existent,
        returns fixed_cols, if not will return first columnname.
        
        INPUT:
            fixed_cols - (str/list of str) columns to fix on shifting
        OUTPUT:
            fixed_cols - (str/list of str) if all columns in fixed_cols exists in chart_df
                         they will be returned, if not the first columnname of chart_df will be returned
        '''
        if not len(set(fixed_cols).intersection(set(self.chart_df.columns))) == len(fixed_cols):
            print("This Column {} doesn't exist in chart_df, using first column instead".format(fixed_cols))
            return [self.chart_df.columns[0]]
        else:
            return fixed_cols
    
    
    def get_shift_cols(self, ext_cols=""):
        '''
        Get the columns that have to be shifted.
        
        INPUT:
            ext_cols - (list of str) Using not all chart_df columns for shifting
        OUTPUT:
            cols - (list of str) Columns to shift with removed fixed columns
        '''
        if ext_cols:
            cols = ext_cols
        else:
            cols = list(self.chart_df.columns)
        for fix_col in self._fixed_cols:
            cols.remove(fix_col)
        
        return cols
    
        
    def single_shift(self, shift_val=-1, ext_cols=""):
        '''
        
        
        INPUT:
            shift_val - (int) day-based level of shifting
            ext_cols - (list of str) Using not all columns for shifting

        OUTPUT:
            df - (DataFrame) with fixed and timeshifted columns
        '''
        cols = self.get_shift_cols(ext_cols)
        df = self.chart_df[self.fixed_cols]
        for col in cols:
            df[col+"_"+str(shift_val)] = self.chart_df[col].shift(shift_val)
        
        return df
    
    
    def gen_multi_shift(self, shift_arr=[ 2, 1, 0, -1, -2]):
        '''
        Generator that returns one shift per iteration from range in shift_arr.
        
        INPUT:
            shift_arr - (list of int) Range for generator based shifting
        OUTPUT:
            (shift_val, df) - Generator(int, DataFrame) shift value and DataFrame with shifted
                              and fixed columns
        '''
        cols = self.get_shift_cols()
        
        df = self.chart_df[self.fixed_cols]
        for shift_val in shift_arr:
            for col in cols:
                df[col] = self.chart_df[col].shift(shift_val)
            yield shift_val, df
            
            
    @staticmethod
    def get_most_causal_cols(df, past=""):
        '''
        Prepares list of columns to return after shifting 
        as DataFrame becomes really big. Holds some static feature sets
        that where identified by previous analysis.
        
        INPUT:
            df - (DataFrame) with shifted columns
            past - (str) returning all past shifts or only week/month specific
        OUTPUT:
            df - (DataFrame) with selected shifted columns
        '''
        
        # opt cols from Granger Causality
        opt_cols = ['bitcoin_Price', 'bitcoin_High', 'bitcoin_Google_Trends_prev_month',
                    'bitcoin_Google_Trends_prev_week', 'alibaba_High_prev_week',
                    'alibaba_Price_prev_week', 'bitcoin_Low_prev_month',
                    'bitcoin_Low_prev_week', 'bitcoin_High_prev_month',
                    'bitcoin_High_prev_week', 'cryptocurrency_Google_Trends_prev_week',
                    'bitcoin_Price_prev_month', 'bitcoin_Price_prev_week','cryptocurrency_Google_Trends',
                    'bitcoin_Low', 'alibaba_Price',
                    'alibaba_High', 'alibaba_Low',
                    'cryptocurrency_Google_Trends_prev_month', 'bitcoin_Google_Trends',
                    'alibaba_Low_prev_week', 'amazon_Price', 'month-1', 'month-2',
                    'alibaba_Low_prev_month', 'amazon_High',
                    'alibaba_Price_prev_month', 'alibaba_High_prev_month',
                    'amazon_High_prev_month', 'amazon_Low_prev_week',
                    'amazon_Price_prev_week', 'amazon_High_prev_week', 'sp500_High',
                    'amazon_Low', 'googl_Price', 'economy_pos_sents_prev_week', 'economy_pos_sents_prev_month']
        
        # opt cols from Feature Optimization
        arimax_opt_cols = [
                   'bitcoin_Price_prev_week',
                   'bitcoin_Price_prev_month',
                   'alibaba_Price_prev_week',
                   'googl_Price_prev_month',
                   'bitcoin_trends_prev_week',
                   'bitcoin_trends_prev_month',
                   'cryptocurrency_trends_prev_week',
                   'cryptocurrency_trends_prev_month',
                   'month-1', 'month-2',
                   ]
        
        
        # all basic time series in chart_df
        cols =   ['bitcoin_Price',
                 'bitcoin_High',
                 'bitcoin_Low',
                 'alibaba_Price',
                 'alibaba_High',
                 'alibaba_Low',
                 'sp500_Price',
                 'sp500_High',
                 'sp500_Low',
                 'dax_Price',
                 'dax_High',
                 'dax_Low',
                 'amazon_Price',
                 'amazon_High',
                 'amazon_Low',
                 'googl_Price',
                 'googl_High',
                 'googl_Low',
                 'bitcoin_Google_Trends',
                 'cryptocurrency_Google_Trends',
                 'economy_pos_sents']
        
        week_cols = ["{}_prev_week".format(col) for col in cols]
        weeks2_cols = ["{}_prev_2weeks".format(col) for col in cols]
        month_cols = ["{}_prev_month".format(col) for col in cols]
        
        return_cols = ["month-1", "month-2"]
        return_cols.extend(cols)
        
        if past=="now":
            return df[cols]
        elif past=="week":
            week_cols.extend(return_cols)
            return df[week_cols]
        elif past=="2weeks":
            weeks2_cols.extend(return_cols)
            return df[week2_cols]
        elif past=="month":
            month_cols.extend(return_cols)
            return df[month_cols]
        elif past=="ari":
            return df[arimax_opt_cols]
        elif past=="all":
            return_cols.extend(week_cols)
            return_cols.extend(weeks2_cols)
            return_cols.extend(month_cols)
            return df[return_cols]
        
        else:
            return df[opt_cols]
    
    
    @staticmethod
    def get_dummy_months(df):
        '''
        Return df with dummy variables for the last three months holding binary 1's and
        0's for beeing in this month or not.
        
        INPUT:
            df - (DataFrame) to add dummy months for
        OUTPUT:
            df - (DataFrame) with added dummy months
        
        '''
        months = df.index.month
        dummy_months = pd.get_dummies(months)
        dummy_months.columns = ['month-%s' % m for m in range(1,len(dummy_months.columns)+1)]
        dummy_months.index = df.index
        
        df = pd.concat([df, dummy_months.iloc[:,:3]], axis=1)
        
        return df
    
    
    @staticmethod
    def get_causal_const_shift(df, past="", zeros="cut"):
        '''
        Returns single train-test split.
        
        INPUT:
            split_factor - (int/float) if int bigger 1 it will take row number for split
                           if float smaller 1 it will take percentage proportion for split
            past - (str) returning all past shifts or only week/month specific
            zeros - (str) the produced zeros after shifting will be "cut" or filled with "zero"
        OUTPUT:
            (train, test) - (DataFrame, DataFrame) splitted train and test datasets
        '''
        df = ShiftChartData.get_dummy_months(df)
        
        causal_cols = ["bitcoin_Price", 
                       "bitcoin_High",
                       "bitcoin_Low",
                       "alibaba_Price",
                       "alibaba_High",
                       "alibaba_Low",
                       "sp500_Price",
                       "sp500_High",
                       "sp500_Low",
                       "dax_Price",
                       "dax_High",
                       "dax_Low",
                       "amazon_Price",
                       "amazon_High",
                       "amazon_Low",
                       "googl_Price",
                       "googl_High",
                       "googl_Low",
                       "bitcoin_Google_Trends",
                       "cryptocurrency_Google_Trends",
                       "economy_pos_sents"]
        try:
            for col in causal_cols:
                if past=="week":
                    df[col+"_prev_week"] = df[col].shift(8)
                elif past=="month":
                    df[col+"_prev_month"] = df[col].shift(31)
                elif past=="2weeks":
                    df[col+"_prev_2weeks"] = df[col].shift(15)
                else:
                    df[col+"_prev_week"] = df[col].shift(8)
                    df[col+"_prev_month"] = df[col].shift(31)
                    df[col+"_prev_2weeks"] = df[col].shift(15)
        except:
            pass
        
        if zeros=="cut" and past=="week":
            df = df.iloc[8:,:]
        elif zeros=="cut" and past=="month":
            df = df.iloc[31:,:]
        elif zeros=="cut" and past=="2weeks":
            df = df.iloc[15:,:]
        elif zeros=="zero":
            df.fillna(0, inplace=True)
        else:
            df = df.iloc[31:, :]
        
        return ShiftChartData.get_most_causal_cols(df, past)
    
    
    
    def return_train_test(self, split_factor=1500, past="all", zeros="cut"):
        '''
        Returns single train-test split.
        
        INPUT:
            split_factor - (int/float) if int bigger 1 it will take row number for split
                           if float smaller 1 it will take percentage proportion for split
            past - (str) returning all past shifts or only week/month specific
            zeros - (str) the produced zeros after shifting will be "cut" or filled with "zero"
        OUTPUT:
            (train, test) - (DataFrame, DataFrame) splitted train and test datasets
        '''
        if split_factor < 1:
            train = self.chart_df[:int(split_factor*(len(self.chart_df)))]
            test = self.chart_df[int(split_factor*(len(self.chart_df))):]
        else: 
            train = self.chart_df[:split_factor]
            test = self.chart_df[split_factor:]
        
        train = ShiftChartData.get_causal_const_shift(train, past=past, zeros=zeros)
        test = ShiftChartData.get_causal_const_shift(test, past=past, zeros=zeros)
        
        return train, test
    
    
    def gen_scaled_train_val_test(self, features, split=""):
        '''
        Generator for returning multiple shifted splits for GRU Cross Validation.
        The default arguments are optimized for one split (main model split) to return.
        
        INPUT:
            features - (list of str) List of features to return split for
            splits - (int) number of total splits
            
        OUTPUT:      
            (train, val, test, split_number) - Generator(DataFrame, DataFrame, DataFrame, int) 
                                               train, validate and test datasets and number of split
        '''
        if split:
            split_range = range(0,split)
        else:
            split_range = range(2,3)
            
        for split_number in split_range:
            train = self.chart_df[:900+(split_number*300)]
            val = self.chart_df[900+(split_number*300):1100+(split_number*300)]
            test = self.chart_df[1100+(split_number*300):]    

            train = ShiftChartData.get_causal_const_shift(train, past="all")[features]
            val = ShiftChartData.get_causal_const_shift(val, past="all")[features]
            test = ShiftChartData.get_causal_const_shift(test, past="all")[features]

            sc = MinMaxScaler()
            train = sc.fit_transform(train)
            sc = MinMaxScaler()
            val = sc.fit_transform(val)
            sc = MinMaxScaler()
            test = sc.fit_transform(test)
            
            yield train, val, test, split_number
            
            
    def return_scaled_test(self, pred, true):
        '''
        Scales prediction and true input time series.
        
        INPUT:
            pred - (DataFrame) Test Data to make predictions for
            true - (DataFrame) True Time Series for comparing
        OUTPUT:
            (pred, pred_c, true, true_c) - (np.array, Scaler Object, np.array, Scaler Object)
                                           Scaled Timeseries and Scaler Objects for later scale inversion
        '''
        pred_c = MinMaxScaler()
        pred = pred_c.fit_transform(pred)
        
        true_c = MinMaxScaler()
        true = true_c.fit_transform(true)
        
        return pred, pred_c, true, true_c
        
        
    def gen_return_splits(self, splits=3, split_size=300, data_len=1800, past="all"):
        '''
        Generator for returning multiple shifted splits for GRU Cross Validation.
        The default arguments are optimized for 3 splits to return.
        
        INPUT:
            splits - (int) number of total splits
            split_size - (int) increment for single split
            data_len - (int) total defined data length for getting complete splits
            past - (str) returning all past shifts or only week/month specific
        OUTPUT:      
            (train, test) - Generator(DataFrame, DataFrame) train and test datasets
        '''
        start_split = int(np.round((data_len/2)/split_size))
        end_split = start_split + splits
        
        
        for i in range(start_split,end_split):
            train = self.chart_df[:i*split_size]
            test = self.chart_df[i*split_size:(i+1)*split_size]
            
            train = ShiftChartData.get_causal_const_shift(train, past=past, zeros="cut")
            test = ShiftChartData.get_causal_const_shift(test, past=past, zeros="cut")
            
            yield train, test

            
class ModelData(ShiftChartData):
    '''
    Expanding ShiftChartData by Model and Forecast specific Functions.
    Generic approach is to use only past data as features/explanatory time series
    for having a time range to predict into future.
    '''
    def __init__(self, 
                 fixed_cols="bitcoin_Price", 
                 window_size=30, 
                 chart_col="Price", 
                 model_path="models",
                 opt_ari_feat=['bitcoin_Google_Trends_prev_month',
                               'cryptocurrency_Google_Trends_prev_month',
                               'alibaba_High_prev_month',
                               'amazon_High_prev_month',
                               'economy_pos_sents_prev_month'],
                 opt_gru_feat =   [ 'bitcoin_Price_prev_month',
                                     'alibaba_Price_prev_month',
                                     'alibaba_High_prev_month',
                                     'amazon_High_prev_month',
                                     'bitcoin_Google_Trends_prev_month',
                                     'economy_pos_sents_prev_month',
                                     'cryptocurrency_Google_Trends_prev_month',
                                                                             ]):
        '''
        Loads the pretrained models, features and will split input data for train and test.
        
        INPUT:
            model_path - (str) Local path where pretrained models are stored
            opt_ari_feat - (list of str) Explanatory Time Series for SARIMAX Model Prediction
            opt_gru_feat - (list of str) Feature Time Series for GRU Model Prediction
        '''
        
        super().__init__(fixed_cols, window_size, chart_col)    
               
        self.arimax_path = str(self.base_path.joinpath("models/sarimax_5_feat_month{}.pkl"))
        self.arimax_model = self.arimax_path.format("")
        self.arimax_split_model = [self.arimax_path.format("_S"+str(i)) for i in range(1,4)]
        self.arimax = pickle.load( open( self.arimax_model, "rb" ) )
        self.opt_ari_feat = opt_ari_feat
        self.train, self.test =  self.return_train_test()
        
        self.gru_path = str(self.base_path.joinpath("models/gru_12_feat_month{}.h5"))
        self.gru_model = self.gru_path.format("")
        self.gru_split_model = [self.gru_path.format("_S"+str(i)) for i in range(0,3)]
        self.gru = load_model(self.gru_model)
        self.gru_timesteps = 5
        self.opt_gru_feat = opt_gru_feat

        
    def get_forecast_dates(self):
        '''
        Returns list of stringified dates for which timespan daily prediction
        is possible.
        
        OUTPUT:
            forecast_list - (list of str) with all days prediction will be available for
        '''
        forecast_exp = self.chart_df[(self.chart_df.index <= self.test.index.max()) & (self.chart_df.index > self.train.index.max())].index[30:]
        return list(forecast_exp.strftime("%Y-%m-%d"))
    
    
    def get_real_price(self, curr_day, shift=-31):
        '''
        Returns real price for given time before "curr_day" and future
        timespan given by "shift".
        
        INPUT:
            curr_day - (str "2019-01-01") day from where to return real price
            shift - (int) number of days to return future price
        OUTPUT:
            (curr_real_price, real_price_31) - (DataFrame, DataFrame) holds past until curr_day
                                               and future real price for shift 
        '''
        offset = shift*-1
        
        curr_date = datetime.strptime(curr_day, "%Y-%m-%d")
        curr_date_offset = curr_date + DateOffset(days=offset)
        
        real_price = self.test[self.test.index <= curr_day][["bitcoin_Price"]]
        curr_real_price = self.apply_boll_bands(df_string="bitcoin_Price", 
                                                price_col="bitcoin_Price",
                                                ext_df=real_price)
        
        real_price_31 = self.test[(self.test.index < curr_date_offset) & (self.test.index >= curr_date)][["bitcoin_Price"]]
        
        return curr_real_price, real_price_31
    
    
    def gru_forecast(self, curr_day, shift=-31):
        '''
        Forecast into future using loaded pretrained GRU model. Will scale before
        and invert scaling after Model prediction.
        
        INPUT:
            curr_day - (str "2019-01-01") day from where to forecast to future
            shift - (int) number of days to look into future
        OUTPUT:
            forecast - (arimax forecast object) forecast holds prediction DataFrame
        '''
        
        forecast_exp = self.prep_forecast(self.opt_gru_feat, curr_day, shift)
       
        sca_fore, fore_tra, sca_price, price_tra = self.return_scaled_test(forecast_exp,
                                                                           forecast_exp[["bitcoin_Price_prev_month"]])
        
        mse, rmse, r2_value,true,predicted = evaluate_model(self.gru, sca_fore, self.gru_timesteps)

        predicted = price_tra.inverse_transform(predicted)
        true = price_tra.inverse_transform(true.reshape(1,-1))
        
        forecast = pd.DataFrame(predicted, index=forecast_exp.iloc[:(self.gru_timesteps)*-1,:].index)
        
        return forecast
        
    
    
    def ari_forecast(self, curr_day, shift=-31):
        '''
        Forecast into future using loaded pretrained SARIMAX model.
        
        INPUT:
            curr_day - (str "2019-01-01") day from where to forecast to future
            shift - (int) number of days to look into future
        OUTPUT:
            forecast - (arimax forecast object) forecast.predicted_mean holds prediction DataFrame
        '''
        forecast_exp = self.prep_forecast(self.opt_ari_feat, curr_day, shift)
                
        forecast = self.arimax.get_forecast(steps=len(forecast_exp), exog=forecast_exp)

        return forecast
    
    
    def prep_forecast(self, features, curr_day, shift):
        '''
        Will prepare time series for forecast by adding offset to testdata
        in dependence of given timeshift. Therefore it's possible to forecast
            - shift=8 days for weekly shifted columns
            - shift=15 days for 2weekly shifted columns
            - shift=31 days for monthly shifted columns
        Any values bigger than that would give knowledge about the future.
            
        INPUT:
            features - (list of str) prepare dataframe with model specific features
            curr_day - (str "2019-01-01") day from where to forecast to future
            shift - (int) number of days to look into future
        OUTPUT:
            forecast_exp - (DataFrame) with appended offset to enable future forecast
        '''
        
        
        feat_prep = [x.replace("_prev_month","{}") for x in features]
        now_feat = [x.format("") for x in feat_prep]
        
        past_df = self.test[self.test.index <= curr_day]
        
        forecast_df = past_df.iloc[shift:,:][now_feat]
                
        forecast_df.index = forecast_df.index + DateOffset(abs(shift))
        
        future_dict = {x.format(""):x.format("_prev_month") for x in feat_prep}
        
        forecast_df.rename(columns=future_dict, inplace=True)
        
        forecast_exp = pd.concat([past_df[features], forecast_df[features]])    
        
        return forecast_exp
         
        
    def cross_validate_arimax(self):
        '''
        Loads pretrained model and returns dict with results for cross validated SARIMAX Model evaluation.
        
        OUTPUT:
            result_dict - (dict) Holds plots and error calculations for each split
        '''
        result_dict = {}
        split_index = 0
        for train, test in self.gen_return_splits():
        
            exog = test[self.opt_ari_feat]
            
            arimax = pickle.load( open( self.arimax_split_model[split_index], "rb" ) )
            forecast = arimax.get_forecast(steps=len(test), exog=exog)
        
            result_dict["S_{}_CORR".format(split_index)] = np.corrcoef(forecast.predicted_mean,test["bitcoin_Price"].values)[0][1]
            result_dict["S_{}_RMSE".format(split_index)] = sqrt(mean_squared_error(forecast.predicted_mean, test["bitcoin_Price"]))
        
            result_dict["S_{}_VALID".format(split_index)] = test["bitcoin_Price"]
            result_dict["S_{}_FORE".format(split_index)] = forecast.predicted_mean
            
            split_index = split_index + 1
                
        return result_dict
        
    def cross_validate_gru(self):
        '''
        Loads pretrained model and returns dict with results for cross validated GRU Model evaluation.
        
        OUTPUT:
            result_dict - (dict) Holds plots and error calculations for each split
        '''
        result_dict = {}
        for train,val,test,split_index in self.gen_scaled_train_val_test(self.opt_gru_feat, split=3):
    
            model = load_model(self.gru_split_model[split_index])
            mse, rmse, r2_value,true,predicted = evaluate_model(model,test,self.gru_timesteps)
            result_dict["S_{}_MSE".format(split_index)] = mse
            result_dict["S_{}_RMSE".format(split_index)] = rmse
            result_dict["S_{}_R2".format(split_index)] = r2_value
            result_dict["S_{}_VALID".format(split_index)] = true
            result_dict["S_{}_FORE".format(split_index)] = predicted.reshape(len(predicted))

        return result_dict
    
    
    def simulate_buy_sell(self, 
                          sim_budget=100000, 
                          min_max_dist=5, 
                          num_neigh=1, 
                          gru_window=12,
                          future_offset_val=31):
        '''
        Simulates Buy and Sells for Bitcoins using GRU model Prediction.
        The daily model forecast will averaged be converted to an percentage based growth index.
        This will flatten some of the false positives in the Model prediction.
        The Buy and Sell triggers are applied by searching local minima and maxima from a daily perspective.
        Some additional rules will prevent false positives from this simple search.
        
        INPUT:
            sim_budget - (int) virtual amount of starting capital
            min_max_dist - (int) minimum distance between minima/maxima in percent for trigger actions
            num_neigh - (int) number of neighbors a minima/maxima needs to be defined
            gru_window - (int) rolling average for daily forecast, higher values mean smoother chart
            future_offset_val - (int) days to use from future forecast
        OUTPUT:
            result_df - (DataFrame) Holds GRU growth chart, real_price and buy/sell triggers
        '''
        budget = sim_budget
        dist_val = min_max_dist
        n = num_neigh
        bitcoin = 0
        last_bitcoin_price=0

        # initial baseline for selling profit
        last_sell_val = budget
        
        # init of n-1 iterations minima and maxima 
        last_sell_max = np.nan
        last_buy_min = np.nan

        # init length of found minima/maxima
        min_val_arr_last = 0
        max_val_arr_last = 0
        
        # start condition for first iteration
        start=0
        last_status = "sell"
        
        # init lists for results
        grow_vals = []
        result_list = []

        for curr_day in self.get_forecast_dates()[5:]:
            result_dict = {}
            # convert date string to datetime
            curr_date = datetime.strptime(curr_day, "%Y-%m-%d")
            result_dict["date"] = curr_date
            
            # apply offset for extracting forecast timespan
            future_offset = curr_date + DateOffset(future_offset_val)

            # get current real_price
            real_price, _ = self.get_real_price(curr_day)
            curr_price = float(real_price[real_price.index == curr_day]["bitcoin_Price"])
            result_dict["curr_price"] = curr_price

            # get GRU model based prediction with maximum of 26 days future forecast
            gru_df = self.gru_forecast(curr_day, shift=-31)
            lookback_offset = 6
            
                
            gru_ext_win = (future_offset_val-lookback_offset)*-1
            gru_future = gru_df[(gru_df.index < future_offset) & (gru_df.index >= curr_date)].iloc[gru_ext_win:].rolling(window=gru_window, min_periods=1).mean()
            # calculate growth for future forecast
            gru_growth = np.round(100 - ((gru_future.values[0]/gru_future.values[-1])*100),2)[0]
            grow_vals.append(gru_growth)
            result_dict["gru_growth"] = gru_growth

            # calculate local minima's from GRU growth plot
            min_val_arr = argrelextrema(np.array(grow_vals), np.less_equal, order=n)[0]
            min_val_arr_len = len(min_val_arr)

            # calculate local maxima's from GRU growth plot
            max_val_arr = argrelextrema(np.array(grow_vals), np.greater, order=n)[0]
            max_val_arr_len = len(max_val_arr)

            # init min and max values that are trigger for buy(min) and sell(max)
            min_val = np.nan
            max_val = np.nan

            # if a new minima was found, a min_val will be set
            if (min_val_arr_len > min_val_arr_last):
                min_val = grow_vals[min_val_arr[-1]]
                min_val_arr_last = min_val_arr_len

            # if a new maxima was found, a max_val will be set
            if (max_val_arr_len > max_val_arr_last):
                max_val = grow_vals[max_val_arr[-1]]
                max_val_arr_last = max_val_arr_last
            
            # Remove all local maxima that are below Zero
            # because this would lead to many wrong selling decisions
            if (max_val < 0):
                max_val = np.nan

            # buying logic: if min_val is set and last action was a sell we will go further
            if np.isfinite(min_val) and (last_status == "sell"):
                # store minimum value for next iteration for calculating distance
                last_buy_min = min_val
                dist = abs(abs(last_buy_min)-abs(last_sell_max))
                # a specific absolute distance has to be between previous maxima current minima
                # or start for first step without dist set
                if (dist>dist_val) or (start==0):
                    # the new minimum has to be smaller than last maximum for
                    # preventing many wrong buying decisions
                    if (min_val < last_sell_max) or (start==0):
                        result_dict["buy_trigger"] = min_val
                        result_dict["buy_price"] = curr_price
                        
                        # negate first iteration condition
                        start=1
                        
                        # buy bitcoin
                        bitcoin = budget/curr_price
                        budget = 0
                        last_status = "buy"

                        #print("{}: Buy Bitcoin for {}\n\tNow having {} Bitcoin\n".format(curr_day, curr_price, bitcoin))

            # selling logic: if max_val is set and last action was a buy we will go further
            if np.isfinite(max_val) and (last_status == "buy"):
                # store maximum value for next iteration for calculating distance
                last_sell_max = max_val
                dist = abs(abs(last_buy_min)-abs(last_sell_max))
                # a specific absolute distance has to be between previous minima current minima
                if (dist>dist_val):
                    result_dict["sell_trigger"] = max_val
                    result_dict["sell_price"] = curr_price
                    
                    # sell bitcoin
                    budget = bitcoin*curr_price
                    bitcoin = 0
                    last_status = "sell"
                    
                    # store 1 if last sell was profitable and 0 if not
                    if budget < last_sell_val:
                        result_dict["profit"] = 0
                    else:
                        result_dict["profit"] = 1
                    last_sell_val = budget
                    #print("{} {}: Sell Bitcoin for {}\n\tNow having {} Dollar\n".format(result_marker ,curr_day, curr_price, budget))

            result_dict["budget"] = budget
            result_dict["bitcoin"] = bitcoin

            result_list.append(result_dict)
        
        result_df = pd.DataFrame(result_list).set_index("date")
        return result_df

    
### APPLY FUNCTIONS ###
    
def convert_values(row, col):
    '''
    Function for pandas per-row apply function. Converts value from specified "col" from
    "5,343.504" to float representation.
    
    INPUT:
        row - (Row of DataFrame) Row with at least one string representation of float value
        col - (str) Name of column with the string representation of float value
    
    OUTPUT:
        val - (float) Returns float value
    '''
    try:
        val = row[col].replace(",","")
    except:
        val = row[col]
    return float(val)

def convert_vol(row, col):
    '''
    Function for pandas per-row apply function. Converts value from specified "col" from
    "10M" or "5K" to float representation.
    
    INPUT:
        row - (Row of DataFrame) Row with at least one string representation of float value
        col - (str) Name of column with the string representation of float value
    
    OUTPUT:
        val - (float) Returns float value
    '''
    letter = row[col][-1]
    val = row[col].rstrip(letter)
    if val == "":
        val = 0
    val = float(val)
    if letter=="M":
        val = val*1000000
    if letter=="K":
        val = val*1000
    
    return val     


### OTHER HELPERS ###

def evaluate_model(model,test,lookback):
    # Function from https://github.com/ninja3697/Stocks-Price-Prediction-using-Multivariate-Analysis/blob/master/Multivatiate-GRU/Multivariate-3-GRU.ipynb
    '''
    Predicts the test data against the fitted GRU model and calculates
    multiple error values for evaluation.
    
    INPUT:
        model - (keras GRU model)
        test - (DataFrame) with the time series test split
        lookback - (int) Timesteps to look into the past
    OUTPUT:
        mse - (float) mean-squared-error between true and prediction chart
        rmse - (float) root-mean-squared-error between true and prediction chart
        r2 - (float) r2 error between true and prediction chart
        Y_test - (np.array) scaled true chart
        Y_hat - (np.array) scaled prediction chart
    '''
    
    X_test = []
    Y_test = []

    # Prepare Test Data from Pandas Series to Numpy compliant Matrix 
    # and split Input DataFrame into Features and Result Vector
    for i in range(lookback,test.shape[0]):
        X_test.append(test[i-lookback:i])
        Y_test.append(test[i][0])
    X_test,Y_test = np.array(X_test),np.array(Y_test)
  

    Y_hat = model.predict(X_test)
    mse = mean_squared_error(Y_test,Y_hat)
    rmse = sqrt(mse)
    r2 = r2_score(Y_test,Y_hat)
    return mse, rmse, r2, Y_test, Y_hat
  