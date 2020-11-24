import pandas as pd
import pathlib
import os
import sys
import numpy as np
import pickle
from utilities import zeeklogreader
from utilities import zeekheader
import pathlib



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
                
        self.df_d = {"conn":pd.DataFrame()}
        self.conn_timestamp = "2020-11-21-18-54-32"
        #self.read_all_data_sources()

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
                self.save_pandas_to_pickle(file_type)


    def update_source(self, file_type=""):
        if file_type in self.data_upd_f.keys():
            if not self.df_d[file_type].empty:
                self.df_d[file_type] = self.df_d[file_type].append(self.parse_json_to_pandas(file_type,update=True), ignore_index=True)                


    def convert_zeek_df(self, file_type=""):
        if file_type in self.data_read_f.keys():
            


    def read_all_data_sources(self):
        for key in self.data_f:
            self.df_d[key] = self.parse_json_to_pandas(key)



 
