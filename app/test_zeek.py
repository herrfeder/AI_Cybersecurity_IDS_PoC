from utilities.parsezeeklogs import ParseZeekLogs
from pygtail import Pygtail
import pathlib
import os
from io import StringIO
import glob
import uuid

# make a UUID based on the host address and current time
uuidOne = uuid.uuid1()


class ZeekLogReader():

    def __init__(self, logoffsets="logoffsets"):
        self.base_path = pathlib.Path(__file__).parent.resolve()
        self.offset_path = os.path.join(self.base_path, "logoffsets")
        self.tmp_path = os.path.join(self.base_path,"logtemp")
        self.log_path_prefix = os.path.join(self.base_path, "../../zeek_input")

        self.conn_log_file = "conn.log"
        self.conn_log = os.path.join(self.log_path_prefix,self.conn_log_file)


        self.conn_fields = ["ts","id.orig_h","id.orig_p",
                            "id.resp_h","id.resp_p","proto",
                            "service","duration","orig_bytes",
                            "resp_bytes","conn_state","local_org",
                            "local_resp","orig_pkts","orig_ip_bytes",
                            "resp_pkts","resp_ip_bytes"]
        self.conn_types = ["time", "string",  "string",
                           "string", "string", "string",
                           "string", "string", "string",
                           "string", "string", "string",
                           "string", "string", "string",
                           "string" ,"string"]


    def read_raw_logs(self, log_file=""):
        print(os.path.join(self.offset_path,self.conn_log_file+".offset"))
        offset_path = os.path.join(self.offset_path,self.conn_log_file+".offset")
        if log_file:
            for line in Pygtail(log_file, offset_file=offset_path):
                if line is not None:
                    yield line


    def read_logs(self, log_file=""):
        if log_file :
            rand_uuid = str(uuid.uuid1())
            temp_file = os.path.join(self.tmp_path,rand_uuid)

            with open(temp_file,"w") as temp_f:
                for log_line in self.read_raw_logs(self.conn_log):
                    temp_f.write(log_line)

            
            #for log_record in ParseZeekLogs(temp_file, output_format="csv", safe_headers=False, fields=self.conn_fields, types=self.conn_types):
            for log_record in ParseZeekLogs(temp_file, output_format="csv", safe_headers=False, fields=["ts","id.orig_h","id.orig_p","id.resp_h","id.resp_p"]):    
                if log_record is not None:
                    print(log_record)
            os.remove(temp_file)

    def update_logs(self):
        self.read_logs(self.conn_log)


if __name__ == "__main__":
    zlr = ZeekLogReader()
    zlr.update_logs()