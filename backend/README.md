# 后端

FastAPI 服务用于加载训练好的 CNN-LSTM 模型，提供预测接口并与 Ryu 控制器交互。

## 主要文件
- `main.py` 接口与业务逻辑
- `CNN_LSTM_Model.py` 模型结构定义
- `utils/` 日志和阈值管理

## 依赖安装
需要 Python 3.8+。可参考下列命令安装常用依赖：
```bash
pip install fastapi uvicorn torch scikit-learn joblib numpy requests
```

## 启动服务
```bash
uvicorn main:app --reload --port 8081
```
