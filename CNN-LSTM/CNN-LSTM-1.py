#%%
# 代码块 1.1

import os
import pandas as pd
from glob import glob
from collections import Counter

# 设置原始数据路径
data_dir = "data/raw/CICDDoS2019"
csv_files = glob(os.path.join(data_dir, "**/*.csv"), recursive=True)

column_counter = Counter()
label_summary = {}

for path in csv_files:
    try:
        df = pd.read_csv(path, nrows=5000)  # 读取前5k行防止内存爆炸
        filename = os.path.basename(path)
        df.columns = [col.strip() for col in df.columns]  # 去除列名空格
        print(f"✅ 文件: {filename}")
        print(f"   ▶️ 行数: {df.shape[0]}, 列数: {df.shape[1]}")
        print(f"   🧱 缺失值总数: {df.isnull().sum().sum()}")

        # 累加完整列名频次
        column_counter.update(df.columns)

        # 标签列统计
        label_col = [col for col in df.columns if 'label' in col.lower() or 'attack' in col.lower()]
        if label_col:
            label_counts = df[label_col[0]].value_counts()
            label_summary[filename] = label_counts.to_dict()
            print(f"   🏷️ 标签列: {label_col[0]}, 分布: {label_counts.to_dict()}")
        else:
            print("   ⚠️ 未找到标签列")
        print("-" * 60)
    except Exception as e:
        print(f"❌ 读取失败: {path}, 原因: {e}")

# 输出所有列名及频次
print("\n📊 所有标准化列名及出现频次（共 {} 个）：".format(len(column_counter)))
for col, count in column_counter.most_common():
    print(f"{col}: {count}")

#%%
# 代码块 1.2

import json

# 排除掉标签列
standard_columns = [col for col in column_counter if col != "Label"]

# 保存为 JSON 文件
with open("feature_columns.json", "w", encoding="utf-8") as f:
    json.dump(standard_columns, f, indent=2)

print(f"✅ 已保存标准字段列 {len(standard_columns)} 项 到 feature_columns.json")

#%%
# 代码块 1.3（最终增强版：优先保留BENIGN）

import numpy as np
from sklearn.preprocessing import MinMaxScaler
from joblib import dump
import os

# 加载标准字段
with open("feature_columns.json", "r") as f:
    standard_columns = json.load(f)

os.makedirs("data/processed", exist_ok=True)
os.makedirs("data/processed_npz", exist_ok=True)

scaler = MinMaxScaler()

for path in csv_files:
    try:
        df = pd.read_csv(path, low_memory=False)
        df.columns = [col.strip() for col in df.columns]

        if "Label" not in df.columns:
            continue

        df["Label"] = df["Label"].astype(str)
        benign_df = df[df["Label"].str.upper() == "BENIGN"]
        attack_df = df[df["Label"].str.upper() != "BENIGN"]

        # 采样策略：全保留 BENIGN + 随机攻击补足
        max_rows = 100000
        keep_benign = benign_df.copy()
        need_attacks = max_rows - len(keep_benign)
        sampled_attack = attack_df.sample(n=need_attacks, random_state=42) if need_attacks > 0 else attack_df.iloc[0:0]

        df_sampled = pd.concat([keep_benign, sampled_attack]).sample(frac=1, random_state=42)

        # 标签编码
        y = df_sampled["Label"].apply(lambda x: 0 if x.upper() == "BENIGN" else 1).values

        # 特征处理
        X = df_sampled.reindex(columns=standard_columns)
        X = X.apply(pd.to_numeric, errors="coerce")
        X = X.replace([np.inf, -np.inf], np.nan).fillna(0)

        # 归一化（第一次拟合）
        X_scaled = scaler.fit_transform(X)

        # 保存
        base = os.path.splitext(os.path.basename(path))[0]
        np.savez_compressed(f"data/processed_npz/{base}.npz", X=X_scaled, y=y)
        pd.DataFrame(X_scaled, columns=standard_columns).to_csv(f"data/processed/{base}.csv", index=False)
        print(f"✅ 清洗+采样: {base} | BENIGN: {len(keep_benign)}, ATTACK: {len(sampled_attack)}, 总: {len(y)}")

    except Exception as e:
        print(f"❌ 错误处理: {path}, 原因: {e}")

# 保存归一化器
dump(scaler, "scaler.pkl")
print("✅ 已保存归一化器 scaler.pkl")

#%%
# 代码块 1.4（修订版：基于 Timestamp 排序滑窗）

import numpy as np
import os
import pandas as pd
from glob import glob

npz_dir = "data/processed_npz"
csv_dir = "data/processed"
out_dir = "data/processed_npz_seq"
os.makedirs(out_dir, exist_ok=True)

window_size = 3
step = 3

npz_files = glob(os.path.join(npz_dir, "*.npz"))

for npz_path in npz_files:
    try:
        base = os.path.splitext(os.path.basename(npz_path))[0]
        csv_path = os.path.join(csv_dir, f"{base}.csv")

        # 载入 CSV 保留时间顺序
        df = pd.read_csv(csv_path)
        df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
        df = df.sort_values("Timestamp").reset_index(drop=True)

        # 同时读取标签
        data = np.load(npz_path)
        y = data["y"]

        # 匹配 Timestamp 排序后的索引（可能丢弃无效时间）
        valid_idx = df["Timestamp"].notnull()
        df = df[valid_idx]
        y = y[valid_idx.values]

        X = df.drop(columns=["Timestamp"]).values  # 去掉时间戳列

        # 滑窗切片
        X_seq, y_seq = [], []
        for i in range(0, len(X) - window_size + 1, step):
            window_y = y[i:i + window_size]
            if np.all(window_y == window_y[0]):
                X_seq.append(X[i:i + window_size])
                y_seq.append(window_y[0])

        np.savez_compressed(f"{out_dir}/{base}_seq.npz", X=np.array(X_seq), y=np.array(y_seq))
        print(f"✅ 时序切片: {base}, 有效序列: {len(y_seq)}")

    except Exception as e:
        print(f"❌ 切片失败: {npz_path}, 原因: {e}")

#%%
# 代码块 1.5（修复：跳过类别不足文件）

import numpy as np
from glob import glob
import os
from sklearn.model_selection import train_test_split

seq_dir = "data/processed_npz_seq"
out_path = "data/final_dataset"
os.makedirs(out_path, exist_ok=True)

X_all, y_all = [], []

for path in glob(os.path.join(seq_dir, "*_seq.npz")):
    try:
        data = np.load(path)
        X, y = data["X"], data["y"]
        if len(np.unique(y)) < 2:
            print(f"⚠️ 跳过: {os.path.basename(path)}，仅包含一个类别")
            continue
        X_all.append(X)
        y_all.append(y)
        print(f"✅ 载入: {os.path.basename(path)}, 序列数: {len(y)}")
    except Exception as e:
        print(f"❌ 加载失败: {path}, 原因: {e}")

X_all = np.concatenate(X_all, axis=0)
y_all = np.concatenate(y_all, axis=0)
print(f"📦 合并后样本数: {len(y_all)}, 输入形状: {X_all.shape}, 标签分布: {np.bincount(y_all)}")

# 划分数据集
X_temp, X_test, y_temp, y_test = train_test_split(X_all, y_all, test_size=0.15, random_state=42, stratify=y_all)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.1765, random_state=42, stratify=y_temp)

# 保存
np.savez_compressed(os.path.join(out_path, "train.npz"), X=X_train, y=y_train)
np.savez_compressed(os.path.join(out_path, "val.npz"), X=X_val, y=y_val)
np.savez_compressed(os.path.join(out_path, "test.npz"), X=X_test, y=y_test)
print("✅ 划分并保存完毕：train/val/test.npz")

#%%
# 代码块 2.1

import torch
import torch.nn as nn

class CNNLSTMClassifier(nn.Module):
    def __init__(self, input_dim=86, hidden_dim=64, lstm_layers=1, dropout=0.3):
        super(CNNLSTMClassifier, self).__init__()

        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=input_dim, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(dropout)
        )

        self.lstm = nn.LSTM(input_size=128, hidden_size=hidden_dim,
                            num_layers=lstm_layers, batch_first=True, bidirectional=True)

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x shape: (B, T, F) → transpose for CNN
        x = x.permute(0, 2, 1)  # (B, F, T)
        x = self.cnn(x)         # (B, 128, T)
        x = x.permute(0, 2, 1)  # (B, T, 128)

        lstm_out, _ = self.lstm(x)     # (B, T, 2*hidden)
        out = lstm_out[:, -1, :]       # 取最后一个时间步
        out = self.classifier(out)     # (B, 1)
        return out.squeeze(1)          # (B,)

# 测试模型输出
model = CNNLSTMClassifier()
x_dummy = torch.randn(8, 3, 86)
y_pred = model(x_dummy)
print("✅ 模型输出形状:", y_pred.shape)  # 应该是 [8]

#%%
# 代码块 2.2

import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

# 配置
BATCH_SIZE = 128
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("✅ 当前设备:", DEVICE)

# 加载数据
def load_dataset(path):
    data = np.load(path)
    X = torch.tensor(data["X"], dtype=torch.float32)
    y = torch.tensor(data["y"], dtype=torch.float32)
    return TensorDataset(X, y)

train_set = load_dataset("data/final_dataset/train.npz")
val_set = load_dataset("data/final_dataset/val.npz")

train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False)

# 初始化模型、优化器、损失函数
model = CNNLSTMClassifier().to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = torch.nn.BCELoss()  # 输出为 Sigmoid 后概率

print(f"✅ 数据加载完成：训练样本 {len(train_set)}，验证样本 {len(val_set)}")

#%%
# 代码块 2.3

import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import os

EPOCHS = 10
BEST_F1 = 0.0
save_path = "model.pt"

for epoch in range(1, EPOCHS + 1):
    model.train()
    train_losses = []
    y_true_train, y_pred_train = [], []

    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()

        train_losses.append(loss.item())
        preds = (outputs >= 0.5).float()
        y_true_train.extend(y_batch.cpu().numpy())
        y_pred_train.extend(preds.cpu().numpy())

    train_acc = accuracy_score(y_true_train, y_pred_train)
    train_f1 = f1_score(y_true_train, y_pred_train)
    print(f"📘 Epoch {epoch} | Train Loss: {np.mean(train_losses):.4f} | Acc: {train_acc:.4f} | F1: {train_f1:.4f}")

    # 评估验证集
    model.eval()
    val_losses = []
    y_true_val, y_pred_val = [], []

    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)

            val_losses.append(loss.item())
            preds = (outputs >= 0.5).float()
            y_true_val.extend(y_batch.cpu().numpy())
            y_pred_val.extend(preds.cpu().numpy())

    val_f1 = f1_score(y_true_val, y_pred_val)
    val_acc = accuracy_score(y_true_val, y_pred_val)
    print(f"🧪 Validation | Loss: {np.mean(val_losses):.4f} | Acc: {val_acc:.4f} | F1: {val_f1:.4f}")

    # 保存最好模型
    if val_f1 > BEST_F1:
        BEST_F1 = val_f1
        torch.save(model.state_dict(), save_path)
        print(f"✅ Saved best model to {save_path}")


#%%
# 代码块 2.4

import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np

# 加载模型
model = CNNLSTMClassifier().to(DEVICE)
model.load_state_dict(torch.load("model.pt"))
model.eval()

# 加载测试集
test_data = np.load("data/final_dataset/test.npz")
X_test = torch.tensor(test_data["X"], dtype=torch.float32).to(DEVICE)
y_test = torch.tensor(test_data["y"], dtype=torch.float32).to(DEVICE)

# 预测
with torch.no_grad():
    outputs = model(X_test)
    preds = (outputs >= 0.5).float()

# 计算准确率与 F1 值
y_true = y_test.cpu().numpy()
y_pred = preds.cpu().numpy()

test_acc = accuracy_score(y_true, y_pred)
test_f1 = f1_score(y_true, y_pred)
test_precision = precision_score(y_true, y_pred)
test_recall = recall_score(y_true, y_pred)

print(f"📊 Test Accuracy: {test_acc:.4f}")
print(f"📊 Test F1: {test_f1:.4f}")
print(f"📊 Test Precision: {test_precision:.4f}")
print(f"📊 Test Recall: {test_recall:.4f}")

#%%
# 代码块 2.5

import os
import random

# 随机选择一个 npz 文件
npz_files = [f for f in os.listdir("data/final_dataset/") if f.endswith(".npz")]
random_file = random.choice(npz_files)

# 加载选定的 npz 文件
print(f"🔄 正在加载文件: {random_file}")
data = np.load(os.path.join("data/final_dataset", random_file))

X_test = torch.tensor(data["X"], dtype=torch.float32).to(DEVICE)
y_test = torch.tensor(data["y"], dtype=torch.float32).to(DEVICE)

# 预测
with torch.no_grad():
    outputs = model(X_test)
    preds = (outputs >= 0.5).float()

# 计算准确率与 F1 值
y_true = y_test.cpu().numpy()
y_pred = preds.cpu().numpy()

test_acc = accuracy_score(y_true, y_pred)
test_f1 = f1_score(y_true, y_pred)
test_precision = precision_score(y_true, y_pred)
test_recall = recall_score(y_true, y_pred)

print(f"📊 Test Accuracy: {test_acc:.4f}")
print(f"📊 Test F1: {test_f1:.4f}")
print(f"📊 Test Precision: {test_precision:.4f}")
print(f"📊 Test Recall: {test_recall:.4f}")

#%%
# 代码块 2.6（修正：从 data/processed_npz_seq 加载）

import random
import numpy as np
import torch
import os

# 从合并前的数据集中随机选择一个 npz 文件
npz_files = [f for f in os.listdir("data/processed_npz_seq") if f.endswith(".npz")]
random_file = random.choice(npz_files)

# 加载选定的 npz 文件
print(f"🔄 正在加载文件: {random_file}")
data = np.load(os.path.join("data/processed_npz_seq", random_file))

X_test = torch.tensor(data["X"], dtype=torch.float32).to(DEVICE)
y_test = torch.tensor(data["y"], dtype=torch.float32).to(DEVICE)

# 预测
with torch.no_grad():
    outputs = model(X_test)
    preds = (outputs >= 0.5).float()

# 计算准确率与 F1 值
y_true = y_test.cpu().numpy()
y_pred = preds.cpu().numpy()

test_acc = accuracy_score(y_true, y_pred)
test_f1 = f1_score(y_true, y_pred)
test_precision = precision_score(y_true, y_pred)
test_recall = recall_score(y_true, y_pred)

print(f"📊 Test Accuracy: {test_acc:.4f}")
print(f"📊 Test F1: {test_f1:.4f}")
print(f"📊 Test Precision: {test_precision:.4f}")
print(f"📊 Test Recall: {test_recall:.4f}")
