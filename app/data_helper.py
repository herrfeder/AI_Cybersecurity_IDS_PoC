import pandas as pd
import pathlib
import os
import sys
import numpy as np
import pickle
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

        self.df_cache_path = os.path.join(self.base_path,"utilities/df_cache")
                
        self.df_d = {"conn":pd.DataFrame(),
                     "temp":""}

        # dirty dirty dirty, hardcoded time offset, beginning of log file
        self.conn_timestamp = "2020-11-21-18-54-32"


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
                self.save_pandas_to_pickle(file_type)




    def update_source(self, file_type=""):
        if file_type in self.data_upd_f.keys():
            if not self.df_d[file_type].empty:
                self.df_d["temp"] = self.parse_json_to_pandas(file_type,update=True)
                self.convert_zeek_df("temp") 
                self.df_d[file_type] = self.df_d[file_type].append(self.df_d["temp"])               


    def convert_zeek_df(self, file_type=""):
        self.convert_epoch_ts(file_type)
        self.sort_set_index(file_type)
        self.drop_unused_conn_fields(file_type)
        self.fill_nan_values(file_type)


    def convert_epoch_ts(self, file_type=""):
        if not isinstance(self.df_d[file_type]["ts"][0],pd.Timestamp):
            epoch = datetime.datetime.strptime('1970-01-01 00:00:00', '%Y-%m-%d %H:%M:%S')
            conn_offset= datetime.datetime.strptime(self.conn_timestamp, '%Y-%m-%d-%H-%M-%S')
            self.df_d[file_type]["ts"] = pd.to_datetime(round(self.df_d[file_type]["ts"]),unit="s") + pd.DateOffset(hours=1)    


    def sort_set_index(self, file_type=""):
        self.df_d[file_type] = self.df_d[file_type].sort_values("ts").set_index("ts") 


    def drop_unused_conn_fields(self, file_type):
        self.df_d[file_type].drop("uid", errors="ignore", inplace=True)
        self.df_d[file_type].drop("tunnel_parents", errors="ignore", inplace=True)
        self.df_d[file_type].drop("local_orig", errors="ignore", inplace=True)
        self.df_d[file_type].drop("local_resp", errors="ignore", inplace=True)

    
    def fill_nan_values(self, file_type=""):
        self.df_d[file_type].fillna("",inplace=True)
 


    def get_timespan_df(self, file_type, time_offset):
        time_delta = self.df_d[file_type].index.max() - datetime.timedelta(minutes=time_offset)

        return self.df_d[file_type][self.df_d[file_type].index > time_delta]


    def get_ten_most_source_ip(self, file_type, time_offset):
        df = self.get_timespan_df(file_type, time_offset)
        return df["id.orig_h"].value_counts()[1:11].to_dict()




 
