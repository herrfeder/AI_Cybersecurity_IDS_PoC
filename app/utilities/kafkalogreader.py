import os
import sys
from kafka import KafkaConsumer
import json
import multiprocessing

try:
    import app.utilities.zeekheader as zeekheader
except BaseException:
    import utilities.zeekheader as zeekheader

def pop_all(l):
    r, l[:] = l[:], []
    return r


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

        self.mp_manager = multiprocessing.Manager()
        self.return_list = self.mp_manager.list()
        
        self.read_process = multiprocessing.Process(target=self.read_logs)


    def init_consumer(self):

        self.consumer = KafkaConsumer(bootstrap_servers=[self.kafka["host"]+":"+self.kafka["port"]])
        self.consumer.subscribe(pattern=self.kafka["topics"])


    def read_logs(self):
       
        for message in self.consumer:
            try:
                log_record = json.loads(message.value)
                self.return_list.append(log_record)
            except:
                pass



    def get_conn_logs(self):
        output_json = pop_all(self.return_list)
        return output_json


if __name__ == "__main__":
    klr = KafkaLogReader()
    klr.read_logs()
