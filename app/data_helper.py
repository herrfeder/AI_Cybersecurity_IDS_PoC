import pandas as pd
import pathlib
import os
import sys
import numpy as np
import pickle
import joblib

try:
    from app.utilities import zeeklogreader
except:
    from utilities import zeeklogreader
try:
    from app.utilities import zeekheader
except:
    from utilities import zeekheader
import pathlib
import datetime
from ipdb import set_trace
import ipinfo



class IDSData():

    def __init__(self):
        self.base_path = pathlib.Path(__file__).parent.resolve()

        self.zlr = zeeklogreader.ZeekLogReader()
        self.data_read_f = {"conn":self.zlr.read_conn_logs, 
                       "dns":"", 
                       "http":""}

        self.data_h = {"conn": zeekheader.conn_fields,
                       "dns":"",
                       "http":""}

        self.data_upd_f = {"conn":self.zlr.update_conn_logs, 
                       "dns":"", 
                       "http":""}

        self.df_d = {"conn":pd.DataFrame(),
                     "temp":""}

        self.model_path = os.path.join(self.base_path, "models")
        self.sup_conn_rf_path = os.path.join(self.model_path, "random_forest.joblib")
        self.sup_conn_rf_model = joblib.load(self.sup_conn_rf_path)
        self.sup_conn_rf_cols = ['duration','orig_pkts','orig_ip_bytes','resp_pkts','resp_ip_bytes']

        self.df_cache_path = os.path.join(self.base_path,"utilities/df_cache")
        self.latlon_cache_path = os.path.join(self.df_cache_path, "latlon.p")


        if os.path.exists(self.latlon_cache_path):
            self.latlon_cache = pickle.load( open( self.latlon_cache_path, "rb" ) )
        else:
            self.latlon_cache = {}


        # dirty dirty dirty SECTION, hardcoded time offset and hardcoded blacklist IPs
        self.conn_timestamp = "2020-11-21-18-54-32"
        self.blacklist_ips = ["94.102.49.191"]


    def parse_json_to_pandas(self, file_type="",update=False):
        if not update:
            json_input = "\n".join(self.data_read_f[file_type]())
        else:
            json_input = "\n".join(self.data_upd_f[file_type]())
        return pd.read_json(json_input, lines=True)


    def read_pickle_to_pandas(self, file_type=""):
        if file_type:
            return pd.read_pickle(os.path.join(self.df_cache_path,file_type+".p"))


    def save_pandas_to_pickle(self, file_type=""):
        if file_type:
            if not self.df_d[file_type].empty:
                self.df_d[file_type].to_pickle(os.path.join(self.df_cache_path,file_type+".p"))


    def read_source(self, file_type="", read_pickle=True):
        if file_type in self.data_read_f.keys():
            if os.path.exists(os.path.join(self.df_cache_path,file_type+".p")) and read_pickle:
                self.df_d[file_type] = self.read_pickle_to_pandas(file_type)
            else:
                self.df_d[file_type] = self.parse_json_to_pandas(file_type)
                self.convert_zeek_df(file_type)
                self.predict_conn_sup_rf(file_type)
                self.save_pandas_to_pickle(file_type)


    def update_source(self, file_type=""):
        if file_type in self.data_upd_f.keys():
            if not self.df_d[file_type].empty:
                self.df_d["temp"] = self.parse_json_to_pandas(file_type,update=True)
                self.convert_zeek_df("temp")
                self.predict_conn_sup_rf("temp")
                self.df_d[file_type] = self.df_d[file_type].append(self.df_d["temp"])               


    ########################
    ###### PREDICTION ######
    ########################

    def predict_conn_sup_rf(self, file_type=""):
        results = self.sup_conn_rf_model.predict(self.df_d[file_type][self.sup_conn_rf_cols])
        self.df_d[file_type]["Prediction_rf"] = results


    ########################
    ###### WRANGLING #######
    ########################

    def convert_zeek_df(self, file_type=""):
        self.convert_epoch_ts(file_type)
        self.sort_set_index(file_type)
        self.drop_unused_conn_fields(file_type)
        self.fill_nan_values(file_type)
        self.clean_duration(file_type)
        self.remove_ips(file_type)


    def convert_epoch_ts(self, file_type=""):
        if not isinstance(self.df_d[file_type]["ts"][0],pd.Timestamp):
            self.df_d[file_type]["ts"] = pd.to_datetime(round(self.df_d[file_type]["ts"]),unit="s") + pd.DateOffset(hours=1)    


    def sort_set_index(self, file_type=""):
        self.df_d[file_type] = self.df_d[file_type].sort_values("ts").set_index("ts")
        self.df_d[file_type]["Time"] = self.df_d[file_type].index 


    def drop_unused_conn_fields(self, file_type):
        self.df_d[file_type].drop("uid", errors="ignore", inplace=True)
        self.df_d[file_type].drop("tunnel_parents", errors="ignore", inplace=True)
        self.df_d[file_type].drop("local_orig", errors="ignore", inplace=True)
        self.df_d[file_type].drop("local_resp", errors="ignore", inplace=True)

    
    def fill_nan_values(self, file_type=""):
        self.df_d[file_type].fillna("",inplace=True)

    
    def clean_duration(self, file_type=""):
        self.df_d[file_type]["duration"] = [0.0 if isinstance(x,str) else x for x in self.df_d[file_type]["duration"]]
 

    def remove_ips(self, file_type=""):
        if self.blacklist_ips:
            for ip in self.blacklist_ips:
                df = self.df_d[file_type]
                self.df_d[file_type] = df[(df["id.orig_h"] != ip) | (df["id.resp_h"] != ip)]


    ###########################
    ###### PLOT RELATED #######
    ###########################

    def get_timespan_df(self, file_type, time_offset):
        time_delta = self.df_d[file_type].index.max() - datetime.timedelta(minutes=time_offset)

        return self.df_d[file_type][self.df_d[file_type].index > time_delta]


    def get_lonlat_from_ip(self, ip):
        if (lonlat_t := self.latlon_cache.get(ip)):
            return lonlat_t
        else:
            handler = ipinfo.getHandler()
            details = handler.getDetails(ip)

            lon = details.longitude.strip("_")
            lat = details.latitude.strip("_")

            self.latlon_cache[ip] = (lon, lat)

            return (lon,lat)

  
    def get_ten_most_source_ip(self, file_type, time_offset):
        df = self.get_timespan_df(file_type, time_offset)
        dict_ip = df["id.orig_h"].value_counts()[1:11].to_dict()

        ips=list(dict_ip.keys())[::-1]
        number=list(dict_ip.values())[::-1]

        return {"ips": ips, "number": number}


    def get_longitude_latitude(self, file_type, time_offset):
        ip_dict = self.get_ten_most_source_ip(file_type, time_offset)
        ips=ip_dict["ips"]
        lon_lat_list = []

        for ip in ips:
            lon_lat_list.append(self.get_lonlat_from_ip(ip))

        pickle.dump( self.latlon_cache, open( self.latlon_cache_path, "wb" ) )

        ip_dict["lonlat"] = lon_lat_list

        return ip_dict

 
