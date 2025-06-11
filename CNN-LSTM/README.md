# 模型训练

该目录包含使用 CNN-LSTM 进行 DDoS 流量识别的训练代码和生成的模型文件。

## 文件说明
- `CNN-LSTM.ipynb` 训练示例 notebook
- `model.pt` 训练得到的模型权重
- `scaler.pkl` 特征标准化器
- `feature_columns.json` 使用的特征列表

## 训练方法
在 Jupyter 环境中运行 `CNN-LSTM.ipynb`，根据提示准备数据并执行各步骤。
训练完成后生成的 `model.pt` 等文件将被后端服务加载用于推理。
