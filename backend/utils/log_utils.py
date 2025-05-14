# utils/log_utils.py

import os
import json
import time

LOG_FILE = "../data/logs.json"

def append_attack_log(entry: dict, limit: int = 100):
    os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)

    try:
        if os.path.exists(LOG_FILE):
            with open(LOG_FILE, "r", encoding="utf-8") as f:
                logs = json.load(f)
        else:
            logs = []

        entry.setdefault("id", int(time.time() * 1000))
        entry.setdefault("timestamp", time.strftime("%Y-%m-%d %H:%M:%S"))

        logs.insert(0, entry)
        logs = logs[:limit]

        with open(LOG_FILE, "w", encoding="utf-8") as f:
            json.dump(logs, f, ensure_ascii=False, indent=2)

    except Exception as e:
        print(f"⚠️ 日志写入失败: {e}")
