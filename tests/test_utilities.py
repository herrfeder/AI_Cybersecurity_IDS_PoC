import pytest
import sys 
sys.path.append('..')
import json
from pygtail import Pygtail
from io import StringIO
import itertools
import re


from app.utilities.parsezeeklogs import ParseZeekLogs
import app.utilities.zeekheader as zeekheader
from app.utilities.zeeklogreader import ZeekLogReader


class TestLogparsing(object):

    @pytest.fixture(autouse=True) 
    def _setup(self):
        self.testlog_path = "tests/testdata/conn.log"
        self.offset_path = "tests/testdata/log_offset"
        self.conn_fields = zeekheader.conn_fields
        self.conn_types = zeekheader.conn_types
        self.seperator = '\t'
        self.set_seperator = [',']
        self.empty_field = ['(empty)']
        self.unset_field = ['-']


    def init_parse_zeek_logs(self, filepath="", fd=""):
        pzl = ParseZeekLogs(filepath=filepath,
                            fd=fd,
                            output_format="json",
                            safe_headers=False,
                            fields=self.conn_fields,
                            types=self.conn_types,
                            seperator=self.seperator,
                            set_seperator=self.set_seperator,
                            empty_field=self.empty_field,
                            unset_field=self.unset_field)

        return pzl

    # parsezeeklogs itself
    def test_read_parsezeeklogs(self):
        pzl = self.init_parse_zeek_logs(filepath=self.testlog_path)

        for log_record in pzl:
            if log_record is not None:
                log_record_json = json.loads(log_record)
                assert log_record_json["ts"] != ""


    # testing zeeklogreader
    def test_read_raw_logs_start_fd(self):
        zlr = ZeekLogReader()

        zlr_rrl = zlr.read_raw_logs(self.testlog_path, start=False)

        for log_record in itertools.islice(zlr_rrl,8):
            if log_record is not None:
                assert log_record.startswith("#")

        for log_record in zlr_rrl:
            if log_record is not None:
                print(log_record)
                assert re.search("^[1-9][1-9]",log_record)


    def test_read_raw_logs_offset_fd(self):
        zlr = ZeekLogReader()

        zlr_rrl = zlr.read_raw_logs(self.testlog_path, start=False, offset_path=self.offset_path)

    
        for log_record in zlr_rrl:
            if log_record is not None:
                print(log_record)
                assert re.search("^[1-9][1-9]",log_record)
