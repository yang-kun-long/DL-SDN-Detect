# ZhiDa 脚手架

## 功能特性

- 🚀 FastAPI + MySQL 后端服务
- 💡 Vue3 + Vite 前端框架
- 🔗 配置好的数据库连接
- 🌐 预置跨域通信支持

## 快速开始

### 1. 克隆仓库

```bash
git clone --branch scaffold --single-branch https://github.com/yang-kun-long/ZhiDa.git
cd ZhiDa
```

### 2. 后端配置

#### 2.1 后端安装依赖

```bash
# 进入后端目录
cd backend

# 创建虚拟环境（Windows）
py -m venv venv
venv\Scripts\activate

# 安装依赖
pip install -r requirements.txt
```

#### 2.2 配置数据库

1. 修改配置文件 backend/config.py ：

```python
DB_CONFIG = {
    "host": "localhost",
    "user": "your_username",
    "password": "your_password",
    "database": "zhida_db",
    "port": 3306
}
```

1. 执行SQL初始化：

```sql
CREATE DATABASE zhida_db CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
CREATE USER 'your_username'@'localhost' IDENTIFIED BY 'your_password';
GRANT ALL PRIVILEGES ON zhida_db.* TO 'your_username'@'localhost';
FLUSH PRIVILEGES;
```

### 3. 前端配置

```bash
# 进入前端目录
cd ../frontend

# 安装依赖
npm install

# 启动开发服务器
npm run dev
```

### 4. 启动服务

服务|启动命令|访问地址
---|---|---
后端api|uvicorn main:app --reload --port 8081|[http://localhost:8000](http://localhost:8000)
前端|npm run dev|[http://localhost:5173](http://localhost:5173)
