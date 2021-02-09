import pytest
import json
from pygtail import Pygtail
from io import StringIO
import itertools
import re
import pandas


import sys 
sys.path.append('..')

from app.data_helper import IDSData


class TestDataHelper(object):

    def test_file_log_to_pandas(self):

        dh = IDSData(file_test=True)
        print(dh.zlr.conn_log)
        dh.read_source(file_type="conn")