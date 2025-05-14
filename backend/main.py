from fastapi import FastAPI,Body
import CNN_LSTM_Model as CNN_LSTM_Model
from fastapi.middleware.cors import CORSMiddleware  # 新增跨域支持
from fastapi.responses import JSONResponse
import httpx
import requests
import numpy as np
import os


from fastapi import FastAPI,HTTPException
from pydantic import BaseModel
import torch
import numpy as np
import json
from joblib import load
import requests

# 假设模型定义在 CNN_LSTM_Model.py 中
from CNN_LSTM_Model import CNNLSTMClassifier
from utils.data_loader import get_flow
from utils.log_utils import append_attack_log
from utils.log_utils import LOG_FILE

app = FastAPI()
THRESHOLD_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "utils", "threshold.json")

# 模型和配置路径
MODEL_PATH = "../CNN-LSTM/model.pt"
SCALER_PATH = "../CNN-LSTM/scaler.pkl"
FEATURE_PATH = "../CNN-LSTM/feature_columns.json"

# 控制器接口地址
RYU_FLOW_ADD_URL = "http://localhost:8080/stats/flowentry/add"
DPID = 1  # 请确保与你的 Mininet 交换机编号一致

# 初始化模型环境
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
with open(FEATURE_PATH, "r") as f:
    feature_columns = json.load(f)
scaler = load(SCALER_PATH)
model = CNNLSTMClassifier(input_dim=len(feature_columns))
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device).eval()

# 请求格式
class PredictInput(BaseModel):
    source_ip: str
    destination_ip: str
    flow: list[list[float]]  # T x F

app = FastAPI()

# 添加CORS中间件（允许前端访问）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

SFLOW_RT_URL = "http://localhost:8008"


@app.get("/")
async def root():
    return {"message": "欢迎使用DDoS防御系统"}


@app.get("/api/topology")
def get_topology():
    switches = requests.get("http://localhost:8080/v1.0/topology/switches").json()
    hosts = requests.get("http://localhost:8080/v1.0/topology/hosts").json()

    topo = {"nodes": [], "links": []}

    # 添加 controller 节点（前端也支持了）
    topo["nodes"].append({"id": "controller", "type": "controller"})

    for sw in switches:
        sw_id = f"s{int(sw['dpid'], 16)}"
        topo["nodes"].append({"id": sw_id, "type": "switch"})
        topo["links"].append({"source": "controller", "target": sw_id, "type": "control"})

    for host in hosts:
        ip = host["ipv4"][0]
        mac = host["mac"]
        port = host["port"]
        dpid = f"s{int(port['dpid'], 16)}"
        hid = f"h{ip.split('.')[-1]}"
        topo["nodes"].append({"id": hid, "type": "host", "ip": ip, "mac": mac})
        topo["links"].append({"source": hid, "target": dpid, "type": "connection"})

    return topo

@app.get("/api/flow_information")
def get_flow_information():
    sflow_url = "http://localhost:8008/activeflows/ALL/mn_flow/json"
    attack_sample = None
    src_ip, dst_ip = None, None
    try:
        # Step 1: 获取源/目标 IP
        resp = requests.get(sflow_url, timeout=3)
        if resp.status_code != 200:
            raise HTTPException(status_code=502, detail="sFlow-RT 请求失败")

        sflow_data = resp.json()
        if not sflow_data:
            raise HTTPException(status_code=428, detail="sFlow-RT 未检测到任何活跃流")

        key = sflow_data[0].get("key", "")
        parts = key.split(",")
        if len(parts) >= 2:
            src_ip, dst_ip = parts[0], parts[1]

        # Step 2: 获取攻击样本
        attack_sample = get_flow()

        if attack_sample is None:
            raise HTTPException(status_code=424, detail="未找到攻击样本（标签为1）")

        return {
            "source_ip": src_ip or "0.0.0.0",
            "destination_ip": dst_ip or "0.0.0.0",
            "flow": attack_sample.tolist()
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"服务器内部错误: {str(e)}")
    
@app.post("/api/predict_from_json")
def predict_from_json(input_data: PredictInput):
    try:
        raw_seq = np.array(input_data.flow)
        if raw_seq.shape[1] != len(feature_columns):
            raise HTTPException(
                status_code=400,
                detail=f"特征维度应为 {len(feature_columns)}, 实际为 {raw_seq.shape[1]}"
            )

        # 标准化
        scaled_seq = scaler.transform(raw_seq)
        input_tensor = torch.tensor(scaled_seq, dtype=torch.float32).unsqueeze(0).to(device)

        # 模型推理
        with torch.no_grad():
            output = model(input_tensor)
            score = output.item()
            threshold = get_current_threshold()
            label = int(score >= threshold)

        # 调用 Ryu 控制器封堵
        if label == 1:
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
                    "nw_src": input_data.source_ip,
                    "nw_dst": input_data.destination_ip
                },
                "actions": []  # 表示 DROP
            }

            try:
                r = requests.post(RYU_FLOW_ADD_URL, json=flow_rule)
                ryu_status = r.status_code
            except Exception as e:
                ryu_status = f"控制器请求失败: {str(e)}"
        else:
            ryu_status = "未触发封堵"

        # 写入日志
        append_attack_log({
            "id": int(time.time() * 1000),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "source_ip": input_data.source_ip,
            "destination_ip": input_data.destination_ip,
            "label": "DDoS攻击" if label == 1 else "正常",
            "score": round(score, 4)
        })

        return {
            "label": label,
            "score": round(score, 4),
            "ryu_action": "blocked" if label == 1 else "none",
            "ryu_status": ryu_status,
            "source_ip": input_data.source_ip,
            "destination_ip": input_data.destination_ip
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
from collections import deque
import time

# 滑动窗口缓存（最多保存10个点）
bps_window = deque(maxlen=10)
pps_window = deque(maxlen=10)
ts_window = deque(maxlen=10)

from collections import deque
import time

# 每次保存最多 10 个数据点用于前端曲线
bps_window = deque(maxlen=10)
pps_window = deque(maxlen=10)
ts_window = deque(maxlen=10)

# 保存上次的累计值和时间，用于速率计算
last_bytes = None
last_packets = None
last_time = None


@app.get("/api/traffic_trend")
def get_traffic_trend():
    global last_bytes, last_time

    try:
        r = requests.get(f"{SFLOW_RT_URL}/analyzer/json", timeout=2)
        r.raise_for_status()
        data = r.json()

        now_ts = time.strftime("%H:%M:%S")
        now = time.time()

        current_bytes = data.get("sFlowBytesReceived", 0)
        cpu_load = data.get("cpuLoadSystem", 0.0)  # float (e.g. 0.031)

        if last_bytes is None or last_time is None:
            mbps = 0.0
        else:
            time_diff = now - last_time
            byte_diff = max(0, current_bytes - last_bytes)
            mbps = round(byte_diff * 8 / time_diff / 1e6, 2)

        # 更新历史
        last_bytes = current_bytes
        last_time = now

        bps_window.append(mbps)
        ts_window.append(now_ts)

        return {
            "timestamps": list(ts_window),
            "bps": list(bps_window),
            "cpu": round(cpu_load * 100, 2)  # percent
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取流量信息失败: {str(e)}")

        
from collections import defaultdict

@app.get("/api/flow_trend")
def get_flow_trend(keys: str = "ipsource,ipdestination", value: str = "bytes"):
    try:
        url = f"{SFLOW_RT_URL}/app/flow-trend/scripts/top.js/flows/json"
        params = {"keys": keys, "value": value}
        res = requests.get(url, params=params, timeout=3)
        res.raise_for_status()
        data = res.json()

        trend = data.get("trend", {})
        topn = trend.get("trends", {}).get("topn", [])
        times = trend.get("times", [])

        series = defaultdict(lambda: [0]*len(times))  # 每条流初始化为 0

        for i, entry in enumerate(topn):
            if isinstance(entry, dict):
                for k, v in entry.items():
                    series[k][i] = round(v / 1_000_000, 3)  # Mbps
            # 否则跳过，比如是 int

        return {
            "timestamps": [time.strftime("%H:%M:%S", time.localtime(t/1000)) for t in times],
            "series": series
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"sFlow-RT 请求失败: {str(e)}")
@app.get("/api/logs")
def get_attack_logs():
    try:
        if not os.path.exists(LOG_FILE):
            return []

        with open(LOG_FILE, "r", encoding="utf-8") as f:
            logs = json.load(f)
        return logs
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"读取日志失败: {str(e)}")
    
@app.post("/api/set_threshold")
def set_threshold(value: float = Body(..., embed=True)):
    if not 0.3 <= value <= 0.9:
        raise HTTPException(status_code=400, detail="阈值必须在 0.3 ~ 0.9 之间")
    try:
        with open(THRESHOLD_CONFIG_PATH, "w", encoding="utf-8") as f:
            json.dump({"threshold": round(value, 2)}, f)
        return {"message": "✅ 阈值已更新", "threshold": round(value, 2)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"无法保存阈值: {str(e)}")
@app.get("/api/get_threshold")
def get_threshold():
    try:
        with open(THRESHOLD_CONFIG_PATH, "r", encoding="utf-8") as f:
            config = json.load(f)
            return {"threshold": config.get("threshold", 0.5)}
    except Exception:
        return {"threshold": 0.5}
