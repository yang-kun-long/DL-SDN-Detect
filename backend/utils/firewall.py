# utils/firewall.py
import requests

RYU_FLOW_ADD_URL = "http://localhost:8080/stats/flowentry/add"
DPID = 1  # 与 Mininet 交换机编号保持一致

def block_ip_flow(source_ip: str, destination_ip: str) -> int | str:
    """封堵指定 IP 流量（添加 drop 流表）"""
    flow_rule = {
        "dpid": DPID,
        "cookie": 1,
        "cookie_mask": 1,
        "table_id": 0,
        "idle_timeout": 30,
        "hard_timeout": 30,
        "priority": 65535,
        "flags": 1,
        "match": {
            "dl_type": 2048,
            "nw_src": source_ip,
            "nw_dst": destination_ip
        },
        "actions": []
    }
    try:
        res = requests.post(RYU_FLOW_ADD_URL, json=flow_rule)
        return res.status_code
    except Exception as e:
        return f"控制器请求失败: {str(e)}"
