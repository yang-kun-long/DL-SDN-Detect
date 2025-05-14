import torch
import numpy as np
import pandas as pd
import json
from joblib import load
from CNN_LSTM_Model import CNNLSTMClassifier  # 确保模型类在这个文件中


# 配置路径
MODEL_PATH = "model.pt"
SCALER_PATH = "scaler.pkl"
FEATURE_PATH = "feature_columns.json"
NPZ_PATH = "data/processed_npz_seq/Syn_seq.npz"  # 你可以换成任何真实路径

# 1. 加载配置文件
with open(FEATURE_PATH) as f:
    feature_columns = json.load(f)

scaler = load(SCALER_PATH)
model = CNNLSTMClassifier(input_dim=len(feature_columns))
model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
model.eval()

# 2. 读取真实攻击样本
data = np.load(NPZ_PATH)
X, y = data["X"], data["y"]

# 3. 找一条攻击样本
attack_sample = None
for xi, label in zip(X, y):
    if label == 1:
        attack_sample = xi
        break

if attack_sample is None:
    print("❌ 没有找到攻击样本。")
    exit()

# 4. 构造 DataFrame 并标准化
df = pd.DataFrame(attack_sample, columns=feature_columns)
scaled = scaler.transform(df)
input_tensor = torch.tensor(scaled, dtype=torch.float32).unsqueeze(0)  # (1, T, F)

# 5. 推理
with torch.no_grad():
    score = model(input_tensor).item()
    label = int(score >= 0.5)

print(f"✅ 模型预测完成：label = {label}, score = {score:.4f}")
