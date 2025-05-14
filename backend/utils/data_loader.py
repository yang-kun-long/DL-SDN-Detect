import os
import numpy as np
import random
from typing import Optional

# 获取项目根路径
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(current_dir))
seq_dir = os.path.join(parent_dir, "CNN-LSTM/data/processed_npz_seq")


def get_flow(seq_dir: str = seq_dir) -> Optional[np.ndarray]:
    """
    随机从本地 .npz 序列数据中读取一条标签为 1 的攻击样本。

    参数:
        seq_dir: 存放 .npz 文件的目录

    返回:
        一个形状为 [T, F] 的攻击样本 ndarray，或 None（若未找到）
    """
    attack_samples = []

    # 随机打乱文件顺序
    files = [f for f in os.listdir(seq_dir) if f.endswith(".npz")]
    random.shuffle(files)

    for file in files:
        data = np.load(os.path.join(seq_dir, file))
        X_seq, y_seq = data["X"], data["y"]
        # 提取所有 y==1 的样本索引
        indices = [i for i, y in enumerate(y_seq) if y == 1]
        if indices:
            random_idx = random.choice(indices)
            return X_seq[random_idx]

    return None
