# necessary for parsing zeek conn.log when not reading from the beginning of the file for updating

conn_fields = [  "ts","uid","id.orig_h",
                "id.orig_p","id.resp_h","id.resp_p",
                "proto","service","duration",
                "orig_bytes","resp_bytes","conn_state",
                "local_orig","local_resp","missed_bytes",
                "history" ,"orig_pkts","orig_ip_bytes", 
                "resp_pkts", "resp_ip_bytes","tunnel_parents"]

conn_types = [  "time", "string", "addr",
                "port", "addr", "port",
                "enum", "string", "interval",
                "count", "count","string", 
                "bool", "bool", "count", 
                "string", "count", "count", 
                "count", "count", "set[string]"]