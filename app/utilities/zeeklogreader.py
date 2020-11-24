try:
    from app.utilities.parsezeeklogs import ParseZeekLogs
except:
    from utilities.parsezeeklogs import ParseZeekLogs
from pygtail import Pygtail
import pathlib
import os
from io import StringIO
try:
    import app.utilities.zeekheader as zeekheader
except:
    import utilities.zeekheader as zeekheader

class ZeekLogReader():

    def __init__(self, logoffsets="logoffsets"):
        self.base_path = pathlib.Path(__file__).parent.resolve()
        self.offset_path = os.path.join(self.base_path, "logoffsets")
        self.tmp_path = os.path.join(self.base_path,"logtemp")
        self.log_path_prefix = os.path.join(self.base_path, "../../../zeek_input")

        self.conn_log_file = "conn.log"
        self.conn_log = os.path.join(self.log_path_prefix,self.conn_log_file)


        # Define these fields for conn.log format
        # It's necessary to read Zeek.logs not from the beginning without headers
        
        self.conn_fields = zeekheader.conn_fields
        self.conn_types = zeekheader.conn_types

        # These fields should be the same for all logfiles
        self.seperator = '\t'
        self.set_seperator = [',']
        self.empty_field = ['(empty)']
        self.unset_field = ['-']  


    def read_raw_logs(self, log_file="", start=False):
        offset_path = os.path.join(self.offset_path,self.conn_log_file+".offset")
        if start:
            if os.path.exists(offset_path):
                os.remove(offset_path)
        if log_file:
            for line in Pygtail(log_file, offset_file=offset_path):
                if line is not None:
                    yield line


    def read_logs(self, log_file="", start=False):
        if log_file :
            raw_log_output = StringIO()

            # optimally I would read the logs line by line into the parser but for first time without any offset
            for log_line in self.read_raw_logs(self.conn_log, start):
                raw_log_output.write(log_line)
            raw_log_output.seek(0)

            for log_record in ParseZeekLogs(fd=raw_log_output, output_format="json", safe_headers=False, 
                                            fields=self.conn_fields, types=self.conn_types, seperator=self.seperator,
                                            set_seperator=self.set_seperator, empty_field=self.empty_field, unset_field=self.unset_field):    
                if log_record is not None:
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
    zlr.update_logs()