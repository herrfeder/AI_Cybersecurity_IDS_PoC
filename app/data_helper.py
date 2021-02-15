import pandas as pd
import pathlib
import os
import sys
import numpy as np
import pickle
import joblib
from autokeras import StructuredDataClassifier
import tensorflow as tf
import keras
import numpy as np
from numpy import asarray
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest

try:
    from app.utilities import kafkalogreader
except BaseException:
    from utilities import kafkalogreader
try:
    from app.utilities import zeekheader
except BaseException:
    from utilities import zeekheader
import pathlib
import datetime
import ipinfo


class IDSData():
    '''
    IDSData manages the input processing and updating of the JSON Zeek Logs and the modelling based on these logs, moreover:
            - LOGPARSING Read and update JSON-based logs from utilities/parsezeeklogs and write into DataFrame
            - PREDICTION Predict with pretrained Models and Anomaly Detection from Webapp Attacks
            - WRANGLING Data Preprocessing > convert time, remove unused columns, blacklist IPs
            - PLOTRELATED Aggregating data for plots

    TODO: The dirty dirty dirty section needs proper functionality and no hardcoded strings
    '''

    def __init__(self, file_test=False):
        self.base_path = pathlib.Path(__file__).parent.resolve()

        self.klr = kafkalogreader.KafkaLogReader(file_test)
        
        self.data_read_k = {"conn": self.klr.get_conn_logs,
                            "dns": "",
                            "http": ""}


        self.data_h = {"conn": zeekheader.conn_fields,
                       "dns": "",
                       "http": ""}


        self.df_d = {"conn": pd.DataFrame(),
                     "temp": ""}

        self.model_path = os.path.join(self.base_path, "models")
        if not os.path.exists(self.model_path):
            os.mkdir(self.model_path)

        if not file_test:
            # RANDOM FOREST
            self.sup_conn_rf_path = os.path.join(
                self.model_path, "RandomForestClassifier_HP_opti.joblib")
            if not os.path.exists(self.sup_conn_rf_path):
                print(
                    "You need to place the Model Files and Folders into the correct locations")
                sys.exit()

            self.sup_conn_rf_model = joblib.load(self.sup_conn_rf_path)
            self.sup_conn_rf_cols = ['duration', 'orig_pkts',
                                    'orig_ip_bytes', 'resp_pkts', 'resp_ip_bytes']

            # NEURAL NET
            self.conn_nn_path = os.path.join(self.model_path, "nn_2_2")
            if not os.path.exists(self.conn_nn_path):
                print(
                    "You need to place the Model Files and Folders into the correct locations")
                sys.exit()

            self.nn_conn_model = keras.models.load_model(self.conn_nn_path)
            self.sup_conn_nn_cols = ['duration', 'orig_pkts',
                                    'orig_ip_bytes', 'resp_pkts', 'resp_ip_bytes']

        # CACHES
        # DataFrame Cache
        self.df_cache_path = os.path.join(self.base_path, "utilities/df_cache")
        if not os.path.exists(self.df_cache_path):
            os.mkdir(self.df_cache_path)
        # IP to LongLat Cache because of ipinfo ratelimiting
        self.latlon_cache_path = os.path.join(self.df_cache_path, "latlon.p")
        if os.path.exists(self.latlon_cache_path):
            self.latlon_cache = pickle.load(open(self.latlon_cache_path, "rb"))
        else:
            self.latlon_cache = {}

        #self.conn_timestamp = self.klr.return_conn_timestamp()

        # dirty dirty dirty SECTION

        # Blacklist IPs should be an functionality in the web app
        self.blacklist_ips = ["94.102.49.191"]

    ########################
    ###### LOGPARSING ######
    ########################
    def parse_json_to_pandas(self, file_type=""):
        json_input = "\n".join(self.data_read_k[file_type]())
        return pd.read_json(json_input, lines=True)

    def parse_dict_to_pandas(self, file_type=""):
        return pd.DataFrame(self.data_read_k[file_type]())

    def read_pickle_to_pandas(self, file_type=""):
        if file_type:
            return pd.read_pickle(os.path.join(
                self.df_cache_path, file_type + ".p"))

    def save_pandas_to_pickle(self, file_type=""):
        if file_type:
            if not self.df_d[file_type].empty:
                self.df_d[file_type].to_pickle(
                    os.path.join(self.df_cache_path, file_type + ".p"))

    def read_source(self, file_type="", read_pickle=True):
        if file_type in self.data_read_k.keys():
            self.df_d[file_type] = self.parse_dict_to_pandas(file_type)
            self.convert_zeek_df(file_type)
            self.predict_conn_sup_rf(file_type)
            self.save_pandas_to_pickle(file_type)


    ########################
    ###### PREDICTION ######
    ########################

    def predict_conn_sup_rf(self, file_type=""):
        results = self.sup_conn_rf_model.predict(
            self.df_d[file_type][self.sup_conn_rf_cols])
        self.df_d[file_type]["Prediction_rf"] = results

    def predict_conn_nn(self, file_type=""):
        results = self.nn_conn_model.predict(
            self.df_d[file_type][self.sup_conn_rf_cols])
        self.df_d[file_type]["Prediction_nn"] = results

    def train_anomaly_detection(
            self, file_type="", train_offset=24, counter_offset=900):
        # highly experimental

        rng = np.random.RandomState(42)
        time_offset = self.df_d[file_type].index.max(
        ) - pd.DateOffset(hours=train_offset)
        timespan_df = self.df_d[file_type][self.df_d[file_type].index > time_offset]
        grouper_offset = str(counter_offset) + 's'

        train_df = pd.DataFrame()
        train_df["port_5m"] = timespan_df.groupby(pd.Grouper(
            freq=grouper_offset, base=30, label='right'))["id.resp_p"].nunique()
        train_df["duration_5m"] = timespan_df.groupby(pd.Grouper(
            freq=grouper_offset, base=30, label='right'))["duration"].nunique()

        self.anomaly_detection_port_max = train_df["port_5m"].max()
        self.anomaly_detection_duration_max = train_df["duration_5m"].max()

        train_df["port_5m_norm"] = (
            train_df["port_5m"] / train_df["port_5m"].max())
        train_df["duration_5m_norm"] = (
            train_df["duration_5m"] / train_df["duration_5m"].max())

        train_header = ["port_5m_norm", "duration_5m_norm"]
        X_train = train_df[train_header].to_numpy()

        clf = IsolationForest(max_samples=100, random_state=rng)
        clf.fit(X_train)

        self.anomaly_detection_port_duration = clf
        self.anomaly_detection_counter = counter_offset

        xx = yy = np.linspace(0, 1, 100)
        Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)

        return X_train, xx, yy, Z

    def return_last_anomaly_metrics(self, timespan_df):

        ano_metrics = {}
        ano_metrics["uniq_ports"] = timespan_df["id.resp_p"].nunique(
        ) / self.anomaly_detection_port_max
        ano_metrics["duration"] = timespan_df["duration"].nunique(
        ) / self.anomaly_detection_duration_max
        #ano_metrics["uniq_hosts"] = timespan_df["id.orig_h"].mean()
        #ano_metrics["all_requests"] = timespan_df.count()[0]

        return [ano_metrics["uniq_ports"], ano_metrics["duration"]]

    def return_anomaly_prediction(self, df):

        df.reset_index(inplace=True)

        offset = df.ts.max() - pd.DateOffset(seconds=self.anomaly_detection_counter)
        metric_array = [self.return_last_anomaly_metrics(
            df.iloc[:-100 + i][df.iloc[:-100 + i].ts > offset]) for i in range(1, 100)]
        prediction = self.anomaly_detection_port_duration.predict(metric_array)

        return prediction

    ########################
    ###### WRANGLING #######
    ########################

    def convert_zeek_df(self, file_type=""):
        self.convert_iso_time(file_type)
        self.sort_set_index(file_type)
        self.drop_unused_conn_fields(file_type)
        self.fill_nan_values(file_type)
        self.clean_duration(file_type)
        self.remove_ips(file_type)

    def convert_iso_time(self, file_type=""):
        if not isinstance(self.df_d[file_type]["ts"][0], pd.Timestamp):
            self.df_d[file_type].loc[:,'ts'] = self.df_d[file_type]['ts'].apply(lambda x: pd.to_datetime(x, format='%Y-%m-%dT%H:%M%:%SZ', errors='ignore'))

    def convert_epoch_ts(self, file_type=""):
        if not isinstance(self.df_d[file_type]["ts"][0], pd.Timestamp):
            self.df_d[file_type]["ts"] = pd.to_datetime(
                round(self.df_d[file_type]["ts"]), unit="s") + pd.DateOffset(hours=1)

    def sort_set_index(self, file_type=""):
        self.df_d[file_type] = self.df_d[file_type].sort_values(
            "ts").set_index("ts")
        self.df_d[file_type]["Time"] = self.df_d[file_type].index

    def drop_unused_conn_fields(self, file_type):
        self.df_d[file_type].drop("uid", errors="ignore", inplace=True)
        self.df_d[file_type].drop(
            "tunnel_parents", errors="ignore", inplace=True)
        self.df_d[file_type].drop("local_orig", errors="ignore", inplace=True)
        self.df_d[file_type].drop("local_resp", errors="ignore", inplace=True)

    def fill_nan_values(self, file_type=""):
        self.df_d[file_type].fillna("", inplace=True)

    def clean_duration(self, file_type=""):
        if not "duration" in self.df_d[file_type].columns:
            self.df_d[file_type]["duration"] = 0.0

        self.df_d[file_type]["duration"] = [0.0 if isinstance(
            x, str) else x for x in self.df_d[file_type]["duration"]]

    def remove_ips(self, file_type=""):
        if self.blacklist_ips:
            for ip in self.blacklist_ips:
                df = self.df_d[file_type]
                self.df_d[file_type] = df[(
                    df["id.orig_h"] != ip) | (df["id.resp_h"] != ip)]

    ###########################
    ###### PLOTRELATED #######
    ###########################

    def get_timespan_df(self, file_type, time_offset):
        time_delta = self.df_d[file_type].index.max(
        ) - datetime.timedelta(seconds=time_offset)

        return self.df_d[file_type][self.df_d[file_type].index > time_delta]

    def get_lonlat_from_ip(self, ip):
        if (lonlat_t := self.latlon_cache.get(ip)):
            return lonlat_t
        else:
            handler = ipinfo.getHandler()
            details = handler.getDetails(ip)

            # have to do this check as with private network IPs details won't
            # hold any info but None
            if not (details.longitude is None):
                lon = details.longitude.strip("_")
                lat = details.latitude.strip("_")

                self.latlon_cache[ip] = (lon, lat)

                return (lon, lat)
            else:
                self.latlon_cache[ip] = (0, 0)
                return (0, 0)

    def get_ten_most_source_ip(self, file_type, time_offset):
        df = self.get_timespan_df(file_type, time_offset * 60)
        dict_ip = df["id.orig_h"].value_counts()[1:11].to_dict()

        ips = list(dict_ip.keys())[::-1]
        number = list(dict_ip.values())[::-1]

        return {"ips": ips, "number": number}

    def get_ten_most_dest_ip(self, file_type, time_offset):
        df = self.get_timespan_df(file_type, time_offset * 60)
        dict_ip = df["id.resp_h"].value_counts()[1:11].to_dict()

        ips = list(dict_ip.keys())[::-1]
        number = list(dict_ip.values())[::-1]

        return {"ips": ips, "number": number}

    def get_longitude_latitude(self, file_type, time_offset):
        ip_dict = self.get_ten_most_source_ip(file_type, time_offset * 60)
        ips = ip_dict["ips"]
        lon_lat_list = []

        for ip in ips:
            lon_lat_list.append(self.get_lonlat_from_ip(ip))

        pickle.dump(self.latlon_cache, open(self.latlon_cache_path, "wb"))

        ip_dict["lonlat"] = lon_lat_list

        return ip_dict
