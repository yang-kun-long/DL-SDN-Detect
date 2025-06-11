# DL-SDN-Detect

该项目提供一个基于深度学习的 SDN DDoS 攻击检测与防御原型。其中包含前端界面、FastAPI 后端服务以及 CNN-LSTM 模型训练脚本。

## 目录结构
- `frontend/`  前端 Vue3 + Vite 应用
- `backend/`   FastAPI 后端，负责推理并与控制器交互
- `CNN-LSTM/`  模型训练代码与权重文件

## 快速开始
1. 按照各目录 README 安装依赖
2. 启动后端：`uvicorn main:app --reload --port 8081`
3. 启动前端：`npm install && npm run dev`

详细使用说明请参阅对应目录下的 README。
