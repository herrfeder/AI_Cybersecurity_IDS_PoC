import pytest
import sys 
sys.path.append('..')
import json

from app.utilities.parsezeeklogs import ParseZeekLogs
import app.utilities.zeekheader as zeekheader
from app.utilities.zeeklogreader import ZeekLogReader


class TestLogparsing(object):

    @pytest.fixture(autouse=True) 
    def _setup(self):
        self.testlog_path = "testdata/conn.log"
        self.conn_fields = zeekheader.conn_fields
        self.conn_types = zeekheader.conn_types
        self.seperator = '\t'
        self.set_seperator = [',']
        self.empty_field = ['(empty)']
        self.unset_field = ['-']


    def test_read_raw_conn_logs_filepath(self):
        pzl = ParseZeekLogs(filepath=self.testlog_path,
                            output_format="json",
                            safe_headers=False,
                            fields=self.conn_fields,
                            types=self.conn_types,
                            seperator=self.seperator,
                            set_seperator=self.set_seperator,
                            empty_field=self.empty_field,
                            unset_field=self.unset_field)

        for log_record in pzl:
            if log_record is not None:
                log_record_json = json.loads(log_record)
                assert log_record_json["ts"] != ""

    def read_raw_logs_fd(self):
        pass


