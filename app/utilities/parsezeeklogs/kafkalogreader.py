try:
    from app.utilities.parsezeeklogs import ParseZeekLogs
except BaseException:
    from utilities.parsezeeklogs import ParseZeekLogs
from pygtail import Pygtail
import pathlib
import os
import sys
from kafka import KafkaConsumer

import linecache
from io import StringIO
try:
    import app.utilities.zeekheader as zeekheader
except BaseException:
    import utilities.zeekheader as zeekheader


class KafkaLogReader():

    def __init__(self, file_test=False):
        
        self.kafka = {}
        try:
            self.kafka["topics"] = os.environ["KAFKA_TOPIC"]
            self.kafka["host"] = os.environ["KAFKA_HOST"]
            self.kafka["port"] = os.environ["KAFKA_PORT"]
        except KeyError:
            print("KAFKA environment variables need to be set")
            sys.exit(1)

        for key in self.kafka.keys():
            if self.kafka[key] == "":
                print("KAFKA environment variables need to be set")
                sys.exit(1)
            elif key=="topics":
                temp_topics = "|".join([x.strip() for x in self.kafka["topics"].split(",")])
                self.kafka["topics"] = temp_topics


        self.init_consumer()

        self.conn_fields = zeekheader.conn_fields
        self.conn_types = zeekheader.conn_types


    def init_consumer(self):

        # To consume latest messages and auto-commit offsets
        self.consumer = KafkaConsumer(bootstrap_servers=[self.kafka["host"]+":"+self.kafka["port"]])
        self.consumer.subscribe(pattern=self.kafka["topics"])


    def read_logs(self, log_file="", start=False):
       
        for log_line in self.consumer:
            log_record = json.loads(log_line)
            yield log_record


    def read_conn_logs(self, init_offset=""):
        output_json = []
        if not init_offset:
            for line in self.read_logs(self.conn_log, start=True):
                output_json.append(line)
        return output_json

    def update_conn_logs(self):
        output_json = []
        for line in self.read_logs(self.conn_log, start=False):
            output_json.append(line)
        return output_json

    def update_all_logs(self):
        self.read_logs(self.conn_log)


if __name__ == "__main__":
    zlr = ZeekLogReader()
    zlr.update_all_logs()
