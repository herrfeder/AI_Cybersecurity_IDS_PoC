import data_helper as dh
from ipdb import set_trace

dh = dh.IDSData()
dh.read_source("conn", read_pickle=False)
set_trace()
dh.update_source("conn")

set_trace()