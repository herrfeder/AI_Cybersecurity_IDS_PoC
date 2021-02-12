import os
import sys
from kafka import KafkaConsumer
import json

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


    def read_logs(self):
       
        for message in self.consumer:
            log_record = json.loads(message.value)
            yield log_record


    def read_conn_logs(self):
        output_json = []
        for line in self.read_logs():
            output_json.append(line)
        return output_json


if __name__ == "__main__":
    klr = KafkaLogReader()
    klr.read_logs()
