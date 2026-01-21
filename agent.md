# FaceRecAPI_DEV 项目文档

## 项目概述

**FaceRecAPI_DEV** 是一个基于深度学习的人脸识别 API 系统，提供了完整的人脸检测、特征提取、相似度匹配和人物库管理功能。

### 核心特性

- 🚀 **高性能架构**: 异步 FastAPI + InsightFace GPU 加速 + 智能多进程处理
- 🎯 **双阶段识别**: 优先候选匹配 + 全局精确匹配，提升识别准确率
- 🔥 **三重混合引擎**: InsightFace 0.7.3 (检测对齐) + FastDeploy ArcFace (特征提取) + dlib (智能回退)
- 🛡️ **智能引擎选择**: GPU 优先 InsightFace，自动回退到 dlib，确保系统稳定性
- 💾 **MongoDB 存储**: NoSQL 数据库存储人物信息和特征向量
- 🌐 **Web UI 界面**: 响应式前端，支持在线识别和人物库管理
- ⚙️ **高可配置性**: TOML 配置文件，灵活调整算法参数
- 📊 **生产就绪**: 支持 Gunicorn/Uvicorn 多进程部署 + Redis 缓存优化

---

## 技术架构

### 技术栈

| 层级 | 技术 | 版本/说明 |
|------|------|---------|
| **Web 框架** | FastAPI + Uvicorn | 异步 Web 服务器 |
| **数据库** | MongoDB (Motor 异步驱动) | NoSQL 数据存储 |
| **缓存系统** | Redis | 特征向量缓存优化 |
| **人脸识别** | InsightFace 0.7.3 | GPU 加速人脸检测和对齐引擎 |
| **深度学习** | FastDeploy-GPU 1.0.7 + ONNX Runtime GPU 1.16.3 | GPU 加速推理引擎 |
| **备用检测** | dlib | CPU 端人脸检测和关键点定位 (智能回退) |
| **图像处理** | OpenCV + NumPy | 图像处理和矩阵计算 |
| **前端** | Bootstrap 5 + Vanilla JS | 响应式 Web UI |
| **配置管理** | TOML | 配置文件格式 |

### 核心模型

- **InsightFace Buffalo_l**: 人脸检测和对齐流水线
  - `det_10g.onnx`: 人脸检测模型 (10G 参数)
  - `w600k_r50.onnx`: 人脸识别模型 (ResNet-50, 600K 训练数据)
  - `1k3d68.onnx`: 3D 人脸对齐模型
  - `2d106det.onnx`: 2D 关键点检测模型 (106 点)
  - `genderage.onnx`: 年龄性别估计模型
- **FastDeploy ArcFace**: `ms1mv3_arcface_r100.onnx` (512 维特征向量提取, 主要)
- **dlib 关键点模型**: `shape_predictor_68_face_landmarks.dat` (68 点关键点, 备用)

### 系统架构图

```
┌─────────────────────────────────────────────────────────┐
│                     客户端 (浏览器/API)                    │
└───────────────────┬─────────────────────────────────────┘
                    │ HTTP/HTTPS
┌───────────────────▼─────────────────────────────────────┐
│               FastAPI 应用层 (main.py)                   │
│  ┌────────────┬────────────┬────────────┬─────────────┐ │
│  │ Web UI 路由│ 人脸识别路由│ 人物管理路由│ 静态资源服务 │ │
│  └────────────┴────────────┴────────────┴─────────────┘ │
└───────────────────┬─────────────────────────────────────┘
                    │
┌───────────────────▼─────────────────────────────────────┐
│                  业务逻辑层 (services/)                   │
│  ┌─────────────────────────────────────────────────────┐│
│  │  PersonService: CRUD 操作、查询优化                   ││
│  └─────────────────────────────────────────────────────┘│
└───────────────────┬─────────────────────────────────────┘
                    │
        ┌───────────┴───────────┬─────────────┐
        │                       │             │
┌───────▼──────────┐   ┌────────▼────────┐   ┌▼─────────┐
│  AI 引擎 (核心)   │   │  MongoDB 数据库  │   │ Redis 缓存│
│ ┌──────────────┐ │   │ ┌──────────────┐│   │ ┌───────┐│
│ │InsightFace   │ │   │ │persons 集合  ││   │ │特征向量 ││
│ │GPU 检测对齐   │ │   │ │- 人物信息     ││   │ │缓存    ││
│ │(Buffalo_l)   │ │   │ │- 特征向量     ││   │ │(Pickle)││
│ └──────────────┘ │   │ │- 照片路径     ││   │ └───────┘│
│ ┌──────────────┐ │   │ └──────────────┘│   └──────────┘
│ │FastDeploy    │ │   └─────────────────┘
│ │ArcFace 特征  │ │
│ └──────────────┘ │
│ ┌──────────────┐ │
│ │dlib 备用引擎 │ │
│ │(智能回退)     │ │
│ └──────────────┘ │
│ ┌──────────────┐ │
│ │特征匹配引擎  │ │
│ │(双阶段匹配)  │ │
│ └──────────────┘ │
└──────────────────┘
```

---

## 目录结构

```
/root/workspace/FaceRecAPI_DEV/
├── app/                                # 主应用目录
│   ├── main.py                         # FastAPI 应用入口 (启动配置)
│   ├── config.toml                     # 项目配置文件 ⚙️
│   ├── requirements.txt                # Python 依赖列表
│   │
│   ├── core/                           # 核心模块
│   │   ├── config.py                  # TOML 配置加载器
│   │   ├── database.py                # MongoDB 连接管理
│   │   ├── ai_engine.py               # AI 引擎核心 (InsightFace + dlib 混合)
│   │   ├── exceptions.py              # 自定义异常类
│   │   └── logger.py                  # 日志系统配置
│   │
│   ├── models/                         # 数据模型 (Pydantic)
│   │   ├── schemas.py                 # 基础数据模型
│   │   ├── request/                   # 请求模型
│   │   │   ├── face_interface_req.py
│   │   │   └── person_interface_req.py
│   │   └── response/                  # 响应模型
│   │       ├── face_interface_rep.py
│   │       └── person_interface_rep.py
│   │
│   ├── router/                         # API 路由
│   │   ├── faces.py                   # 人脸识别 API (/recognize)
│   │   ├── persons.py                 # 人物管理 API (/persons)
│   │   ├── ops.py                     # 运维与监控接口 (/ops)
│   │   └── web.py                     # Web UI 路由 (/)
│   │
│   ├── services/                       # 业务逻辑层
│   │   ├── cache_service.py           # Redis 缓存与特征加载
│   │   ├── face_service.py            # 识别流程封装（当前主要由路由直接调用 ai_engine）
│   │   ├── ops_stats.py               # API 统计聚合
│   │   └── person.py                  # 人物 CRUD 服务
│   │
│   ├── middleware/                     # Starlette/FastAPI 中间件
│   │   └── api_stats_middleware.py    # API 调用统计 & TTL 清理
│   │
│   ├── utils/                          # 工具函数
│   │   ├── image_loader.py            # 图像加载、尺寸校验、ROI
│   │   └── utils_mongo.py             # MongoDB 文档转换
│   │
│   ├── static/                         # Web 前端资源 (Bootstrap + JS)
│   │   ├── templates/index.html
│   │   └── js/app.js
│   │
│   ├── media/                          # 媒体存储
│   │   └── person_photos/             # 已裁剪的头像
│   │
│   ├── logs/                           # 日志文件夹（uvicorn/app 日志）
│   │
│   ├── ai_models/                      # AI 模型文件目录
│   │   ├── README.md                  # 模型放置说明
│   │   ├── models/buffalo_l/...       # InsightFace 模型套件
│   │   ├── ms1mv3_arcface_r100.onnx   # ArcFace 特征提取模型
│   │   └── shape_predictor_68_face_landmarks.dat  # dlib 68 点模型
│   │
│   ├── tests/                          # 测试与脚本
│   │   ├── integration/
│   │   ├── unit/
│   │   ├── stress_test_n.py           # 压测脚本
│   │   └── test_recognize.py          # 接口验证脚本
│
└── agent.md                            # 本文档 📄
```

---

## 核心功能

### 1. 人脸识别流程

```
输入图片 (Base64/文件)
    │
    ▼
┌────────────────────┐
│ 1. 图像预处理      │ ← image_loader.py (图像验证、尺寸检查)
└────────┬───────────┘
         │
         ▼
┌────────────────────┐
│ 2. 智能引擎选择    │ ← ai_engine.py (InsightFace 优先 + dlib 回退)
│   - 检查 GPU 可用性│
│   - 选择最优引擎   │
└────────┬───────────┘
         │
         ▼ (InsightFace 路径 - 主流程)
┌────────────────────┐
│ 3. InsightFace     │ ← Buffalo_l 模型套件 (GPU 加速)
│   检测和对齐       │
│   - 人脸检测       │   输出: 人脸包围框 + 关键点
│   - 自动对齐       │   输出: 对齐后人脸 (112x112)
└────────┬───────────┘
         │
         ▼
┌────────────────────┐
│ 4. FastDeploy      │ ← ArcFace R100 模型 (GPU 加速)
│   特征提取         │
│   - ONNX 推理      │   输出: 512 维特征向量
│   - L2 归一化      │
└────────┬───────────┘
         │
         ▼ (dlib 路径 - 备用流程)
┌────────────────────┐
│ 3b. dlib 检测      │ ← 传统多步骤处理 (CPU)
│   - 灰度化         │
│   - 人脸检测       │   输出: 人脸包围框
│   - 68 点关键点    │
│   - 5 点对齐       │   输出: 对齐后人脸 (112x112)
│   - FastDeploy特征 │   输出: 512 维特征向量
└────────┬───────────┘
         │
         ▼
┌────────────────────┐
│ 5. 双阶段特征匹配  │
│ ┌────────────────┐ │
│ │阶段 1: 候选匹配│ │ ← targets 优先匹配 (阈值 0.2, 宽松)
│ └────────────────┘ │
│ ┌────────────────┐ │
│ │阶段 2: 全局匹配│ │ ← 全库匹配 (阈值 0.4, 严格)
│ └────────────────┘ │   + Redis 缓存优化
└────────┬───────────┘
         │
         ▼
识别结果 (相似度、匹配人物、包围框)
```

### 2. InsightFace vs dlib 对比

| 特性 | InsightFace + FastDeploy (主引擎) | dlib + FastDeploy (备用引擎) |
|------|----------------------------------|------------------------------|
| **处理方式** | 检测对齐 + 特征提取分离 | 多步骤流水线处理 |
| **计算设备** | GPU 加速 (CUDA) | CPU 处算 |
| **检测精度** | 更高 (SOTA 模型) | 较高 (传统方法) |
| **处理速度** | 更快 (GPU 并行) | 较慢 (CPU 串行) |
| **资源占用** | GPU 显存 | CPU 内存 |
| **模型大小** | 较大 (~500MB) | 较小 (~100MB) |
| **适用场景** | 高并发生产环境 | 资源受限环境 |

### 3. 双阶段识别策略

**为什么需要双阶段？**
- **阶段 1 (候选匹配)**: 快速匹配重点关注的人物 (targets 列表)，阈值较低 (0.2) 提升召回率
- **阶段 2 (全局匹配)**: 严格匹配数据库所有人物，阈值较高 (0.4) 降低误识别

**实现逻辑** (见 `router/faces.py`)：
```python
# 阶段 1: 优先匹配候选人物 (targets)
if targets:
    candidate_threshold = settings.face.candidate_threshold  # 0.2
    similarity, matched_person = find_best_match(
        embedding, targets_embeddings, threshold=candidate_threshold
    )
    if similarity > candidate_threshold:
        return matched_person  # ✅ 候选匹配成功

# 阶段 2: 全局匹配
threshold = settings.face.threshold  # 0.4
all_embeddings = await get_all_embeddings_from_db(db)
similarity, matched_person = find_best_match(
    embedding, all_embeddings, threshold=threshold
)
return matched_person if similarity > threshold else None
```

### 4. 特征向量管理

**特征向量规格**:
- **维度**: 512
- **数据类型**: float32
- **归一化**: L2 归一化 (||embedding|| = 1)
- **相似度计算**: 余弦相似度 = dot(emb_a, emb_b)
- **相似度范围**: [0, 1] (越接近 1 越相似)

**生成方式**:
- **InsightFace 路径**: InsightFace 检测对齐 → FastDeploy ArcFace 提取 512 维特征向量
- **dlib 路径**: dlib 检测对齐 → FastDeploy ArcFace 提取 512 维特征向量

**存储方式**:
```python
# MongoDB 二进制存储
embedding_binary = Binary(embedding.astype(np.float32).tobytes())

# 读取时解码
embedding_array = np.frombuffer(doc['embedding'], dtype=np.float32)
```

**缓存优化**:
```python
# Redis 缓存特征向量 (Pickle 序列化，性能提升 5-10 倍)
from app.services.cache_service import cache_service

# 获取所有特征向量（自动缓存管理）
embeddings = await cache_service.get_all_embeddings()

# 手动刷新缓存
await cache_service.reload_all_embeddings()
```

---

## API 接口文档

### 1. 人脸识别 API

#### POST `/recognize`

**功能**: 在线人脸识别(多人脸N:N)

**请求体** (JSON):
```json
{
  "photo": "data:image/jpeg;base64,/9j/4AAQSkZJRg...",
  "targets": ["T001", "T002"],
  "threshold": 0.4,
  "points": [
    {"x": 0, "y": 0},
    {"x": 1920, "y": 0},
    {"x": 1920, "y": 540},
    {"x": 0, "y": 540}
  ]
}
```

**响应** (JSON):
```json
{
  "statusCode": 200,
  "message": "识别成功",
  "data": {
    "hasFace": true,
    "bboxs": [
      {"x": 100, "y": 150, "w": 200, "h": 200},
      {"x": 350, "y": 120, "w": 180, "h": 180}
    ],
    "threshold": 0.4,
    "match": [
      {
        "id": "507f1f77bcf86cd799439011",
        "name": "张三",
        "number": "T001",
        "similarity": "87.45%",
        "is_target": true
      },
      {
        "id": "507f1f77bcf86cd799439012",
        "name": "李四",
        "number": "T002",
        "similarity": "82.10%",
        "is_target": false
      }
    ],
    "message": "检测到2个人脸，匹配2位候选人"
  }
}
```

**错误响应**:
```json
{
  "code": 400,
  "message": "图像格式无效",
  "data": null
}
```

---

### 2. 人物管理 API

#### POST `/persons` - 添加人物

**请求体**:
```json
{
  "name": "李四",
  "number": "2024002",
  "photo": "data:image/jpeg;base64,/9j/4AAQSkZJRg..."
}
```

**响应**:
```json
{
  "code": 200,
  "message": "人物添加成功",
  "data": {
    "id": "65abc123...",
    "name": "李四",
    "number": "2024002",
    "photo_path": "/media/person_photos/2024002.jpg",
    "tip": "人脸质量良好"
  }
}
```

#### GET `/persons` - 获取人物列表

**查询参数**:
- `skip`: 跳过数量 (分页, 默认 0)
- `limit`: 返回数量 (默认 100, 最大 5000)

**响应**:
```json
{
  "code": 200,
  "data": [
    {
      "id": "65abc123...",
      "name": "张三",
      "number": "2024001",
      "photo_path": "/media/person_photos/2024001.jpg"
    }
  ]
}
```

#### GET `/persons/search` - 搜索人物

**查询参数**:
- `name`: 姓名关键词 (可选)
- `number`: 编号 (可选)

**响应**: 同 `/persons`

#### DELETE `/persons/delete` - 删除人物

**请求体**:
```json
{
  "name": "张三",  // 可选
  "number": "2024001"  // 可选
}
```

**响应**:
```json
{
  "code": 200,
  "message": "成功删除 1 个人物"
}
```

#### DELETE `/persons/{person_id}` - 按 ID 删除人物

**路径参数**:
- `person_id`: MongoDB ObjectId

**响应**: 同上

---

### 3. Web UI 路由

#### GET `/`

**功能**: 访问 Web UI 管理界面

**认证**: HTTP Basic Auth
- 用户名: `admin` (默认, 可在 config.toml 修改)
- 密码: `admin`

**界面功能**:
- ✅ 在线人脸识别 (拖拽上传)
- ✅ 人物库管理 (添加、搜索、删除)
- ✅ 批量上传人物照片
- ✅ 实时识别结果展示

---

## 配置文件详解

### config.toml 配置项

```toml
# ====== 数据库配置 ======
[db]
username = "root"               # MongoDB 用户名
password = "root"               # MongoDB 密码
host = "10.80.5.25"            # MongoDB 地址
port = "27017"                 # MongoDB 端口
database = "facerecapi"        # 数据库名称
auth_source = "admin"          # 认证源
limit = 5000                   # 单次查询最大结果数

# ====== Redis 缓存配置 ======
[redis]
host = "localhost"             # Redis 服务器地址
port = 6379                    # Redis 端口
password = ""                  # Redis 密码 (可选)
db = 0                         # Redis 数据库编号
cache_ttl = 300               # 缓存过期时间 (秒)

# ====== 人脸检测引擎配置 ======
[face_detection]
detector = "insightface"       # 主检测引擎: "insightface" | "dlib"
fallback_detector = "dlib"     # 备用检测引擎

# InsightFace 配置
[face_detection.insightface]
model_name = "buffalo_l"       # 模型名称: buffalo_l | buffalo_m | buffalo_s
device = "gpu"                 # 计算设备: "gpu" | "cpu"
gpu_id = 1                     # GPU 设备 ID
det_size = 640                 # 检测尺寸 (像素)
det_thresh = 0.5              # 检测阈值 (0-1)

# dlib 配置 (备用)
[face_detection.dlib]
max_workers = 2                # 进程池工作线程数

# ====== 人脸识别参数 ======
[face]
threshold = 0.4                # 全局匹配阈值 (0-1, 越高越严格)
candidate_threshold = 0.2      # 候选匹配阈值 (优先匹配 targets 列表)
rec_min_face_hw = 50           # 识别阶段最小人脸尺寸 (像素)

# ====== Web UI 登录配置 ======
[frontlogin]
username = "admin"             # Web UI 用户名
password = "admin"             # Web UI 密码

# ====== 图像验证配置 ======
[image]
max_feature_image_width_px = 9999   # 最大图像宽度
max_feature_image_height_px = 9999  # 最大图像高度
min_feature_image_width_px = 80     # 最小图像宽度 ⚠️
min_feature_image_height_px = 80    # 最小图像高度 ⚠️
max_feature_image_size_m = 10       # 最大图像文件大小 (MB)
max_face_hw = 999                   # 最大人脸区域尺寸
min_face_hw = 50                    # 最小人脸区域尺寸 ⚠️

# ====== 日志配置 ======
[logger]
level = "INFO"                      # 日志级别 (DEBUG/INFO/WARNING/ERROR)
filename = "/app/logs/facerecapi.log"  # 日志文件路径
```

### 关键配置调优建议

| 配置项 | 默认值 | 调优建议 |
|--------|--------|---------|
| `face.threshold` | 0.4 | 提高减少误识别，降低提升召回率 |
| `face.candidate_threshold` | 0.2 | 优先匹配阈值，建议保持较低 |
| `face_detection.detector` | insightface | 生产环境推荐 InsightFace，资源受限时用 dlib |
| `face_detection.insightface.gpu_id` | 1 | 多 GPU 环境下选择空闲 GPU |
| `face_detection.insightface.det_size` | 640 | 提高检测精度，但会增加计算量 |
| `face_detection.dlib.max_workers` | 2 | CPU 核心数的 50%-75% (避免内存溢出) |
| `redis.cache_ttl` | 300 | 根据人物库更新频率调整缓存时间 |
| `db.limit` | 5000 | 人物库超过 1 万时建议分页查询 |

---

## 数据库设计

### MongoDB 集合: `persons`

**文档结构**:
```javascript
{
  "_id": ObjectId("65abc123..."),      // MongoDB 主键
  "name": "张三",                       // 人物姓名 (可重复)
  "number": "2024001",                 // 人物编号 (唯一索引) ⭐
  "photo_path": "/media/person_photos/2024001.jpg",  // 照片路径
  "embedding": BinData(...),           // 512 维特征向量 (二进制存储)
  "tip": "人脸质量良好"                  // 人脸质量提示
}
```

**索引设计**:
```javascript
// 唯一性约束 (防止重复录入)
db.persons.createIndex({ "number": 1 }, { unique: true })

// 查询优化索引
db.persons.createIndex({ "name": 1 })
db.persons.createIndex({ "photo_path": 1 })
```

**数据访问模式**:
- 插入: 每次添加人物时插入 1 条文档
- 查询:
  - 识别时全表扫描 (`limit: 5000`) 获取所有特征向量
  - 搜索时通过 `name` 或 `number` 索引查询
- 删除: 按 `_id` 或 `number` 删除，同时删除对应照片文件

---

## 部署指南

### 环境要求

**硬件**:
- CPU: 4 核+ (推荐 8 核)
- 内存: 16GB+ (推荐 32GB, InsightFace 模型较大)
- GPU: NVIDIA GPU (CUDA 11.x+, 显存 6GB+) - 推荐用于 InsightFace
- 存储: 50GB+ (InsightFace 模型 + 人物照片存储)

**软件**:
- Python: 3.8+
- MongoDB: 4.4+
- Redis: 6.0+ (用于特征向量缓存)
- CUDA: 11.x+ (GPU 部署时需要)
- 操作系统: Linux (Ubuntu 20.04+ / CentOS 7+)

### 安装步骤

#### 1. 安装 Python 依赖

```bash
cd /root/workspace/FaceRecAPI_DEV/app
pip install -r requirements.txt
```

**关键依赖**:
```
fastapi
uvicorn[standard]
gunicorn
motor              # MongoDB 异步驱动
insightface==0.7.3 # 人脸检测和对齐引擎
onnxruntime-gpu==1.16.3  # ONNX GPU 推理 (兼容 CUDA 11.6)
fastdeploy-gpu-python==1.0.7  # FastDeploy GPU 推理框架
dlib               # 备用人脸检测
opencv-python      # 图像处理
numpy==1.26.4      # 数值计算 (兼容 onnxruntime-gpu 1.16.3)
redis[hiredis]>=5.0.0  # Redis 缓存系统
```

#### 2. 配置数据库服务

**MongoDB 配置**:
```bash
# 启动 MongoDB 服务
systemctl start mongod

# 创建数据库和用户
mongo
> use facerecapi
> db.createUser({
    user: "root",
    pwd: "root",
    roles: [{ role: "readWrite", db: "facerecapi" }]
  })
```

**Redis 配置**:
```bash
# 启动 Redis 服务
systemctl start redis

# 测试 Redis 连接
redis-cli ping
# 应该返回: PONG
```

#### 3. 修改配置文件

```bash
cd /root/workspace/FaceRecAPI_DEV/app
vim config.toml

# 必须修改的配置项:
# - [db] MongoDB 数据库连接信息
# - [redis] Redis 缓存连接信息
# - [face_detection] 选择检测引擎 (insightface/dlib)
# - [face_detection.insightface] GPU 配置
# - [frontlogin] Web UI 登录密码
```

#### 4. 启动服务

**开发模式** (单进程 + 热重载):
```bash
cd /root/workspace/FaceRecAPI_DEV
PYTHONPATH=/root/workspace/FaceRecAPI_DEV OMP_NUM_THREADS=1 \
  uvicorn app.main:app --host 0.0.0.0 --port 8003 --reload
```

**生产模式** (推荐):
```bash
cd /root/workspace/FaceRecAPI_DEV
PYTHONPATH=/root/workspace/FaceRecAPI_DEV OMP_NUM_THREADS=1 \
  gunicorn -w 1 -k uvicorn.workers.UvicornWorker \
  --bind 0.0.0.0:8003 \
  --timeout 120 \
  --log-level info \
  app.main:app
```

**使用 systemd 管理服务**:
```bash
sudo vim /etc/systemd/system/facerecapi.service
```

```ini
[Unit]
Description=FaceRecAPI Service
After=network.target mongod.service

[Service]
Type=notify
User=root
WorkingDirectory=/root/workspace/FaceRecAPI_DEV
Environment="PYTHONPATH=/root/workspace/FaceRecAPI_DEV"
Environment="OMP_NUM_THREADS=1"
ExecStart=/usr/local/bin/gunicorn -w 1 -k uvicorn.workers.UvicornWorker \
          --bind 0.0.0.0:8003 \
          --timeout 120 \
          app.main:app
Restart=on-failure

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl daemon-reload
sudo systemctl enable facerecapi
sudo systemctl start facerecapi
sudo systemctl status facerecapi
```

#### 5. 验证部署

```bash
# 检查服务状态
curl http://localhost:8003/

# 访问 Web UI
# 浏览器打开: http://<服务器IP>:8003
# 输入用户名/密码: admin/admin
```

---

## 开发指南

### 代码结构规范

**目录职责**:
- `core/`: 核心基础设施 (配置、数据库、AI 引擎、日志)
- `models/`: 数据模型定义 (Pydantic schemas)
- `router/`: API 路由和请求处理
- `services/`: 业务逻辑层 (与数据库交互)
- `utils/`: 通用工具函数

**依赖方向**:
```
router/ → services/ → core/ (database, ai_engine)
         ↓
       models/
```

### 添加新的 API 端点

**步骤**:
1. 在 `models/request/` 定义请求模型
2. 在 `models/response/` 定义响应模型
3. 在 `services/` 实现业务逻辑
4. 在 `router/` 添加路由处理函数
5. 在 `main.py` 注册路由

**示例** (添加 `/health` 健康检查):
```python
# router/health.py
from fastapi import APIRouter
from models.response.common import HealthResponse

router = APIRouter(prefix="/health", tags=["health"])

@router.get("/", response_model=HealthResponse)
async def health_check():
    return {"status": "ok", "version": "1.0.0"}

# main.py
from router import health
app.include_router(health.router)
```

### 调试技巧

**1. 查看日志**:
```bash
tail -f /root/workspace/FaceRecAPI_DEV/app/logs/face.log
```

**2. 开启 DEBUG 日志**:
```toml
# config.toml
[logger]
level = "DEBUG"
```

**3. 测试识别功能**:
```bash
cd /root/workspace/FaceRecAPI_DEV/app
python test_recognize.py
```

**4. 压力测试**:
```bash
python stress_test.py
```

### 性能优化建议

**1. 数据库优化**:
- 建立复合索引: `{ "name": 1, "number": 1 }`
- 使用投影查询减少网络传输: `db.persons.find({}, {"embedding": 1})`

**2. Redis 缓存优化**:
```python
# 缓存特征向量
async def get_all_embeddings_cached():
    cache_key = "all_embeddings"
    cached = await redis.get(cache_key)
    if cached:
        return pickle.loads(cached)

    embeddings = await get_all_embeddings_from_db()
    await redis.setex(cache_key, 300, pickle.dumps(embeddings))  # 5 分钟缓存
    return embeddings
```

**3. InsightFace 优化**:
- 使用 GPU 加速: `device = "gpu"`
- 选择合适模型: `buffalo_l` (高精度) vs `buffalo_s` (高速度)
- 批量推理: 多张图片同时处理

**4. 多进程调优**:
- InsightFace GPU 模式: Gunicorn workers = 1 (避免 GPU 资源竞争)
- dlib CPU 模式: `max_workers` = CPU 核心数的 50%-75%
- 混合模式: 根据实际负载动态调整

---

## 常见问题 (FAQ)

### Q1: 如何提高识别准确率？

**A**: 调整以下参数:
1. **提高阈值**: `face.threshold = 0.5` (降低误识别)
2. **提升图像质量**: 要求上传高清正面照片
3. **人脸尺寸**: 确保人脸像素大于 `min_face_hw = 80`

### Q2: 如何支持更大规模人物库 (10 万+)?

**A**: 优化策略:
1. **向量索引**: 使用 Faiss 或 Milvus 向量数据库
2. **分片存储**: MongoDB Sharding
3. **增量更新**: 仅加载新增人物特征

### Q3: GPU 显存不足怎么办？

**A**:
1. 降低 Gunicorn workers 数量为 1
2. 切换到 CPU 模式: `face_detection.insightface.device = "cpu"`
3. 使用较小的 InsightFace 模型: `model_name = "buffalo_s"`
4. 回退到 dlib 引擎: `face_detection.detector = "dlib"`

### Q4: 如何在 CPU 环境下部署？

**A**:
1. 修改配置文件:
   ```toml
   [face_detection.insightface]
   device = "cpu"
   ```
2. 安装 CPU 版本依赖:
   ```bash
   pip install onnxruntime  # 替代 onnxruntime-gpu
   ```
3. 或直接使用 dlib 引擎:
   ```toml
   [face_detection]
   detector = "dlib"
   ```

### Q5: 如何实现多人脸识别？

**A**: 当前已支持！InsightFace 和 dlib 引擎都会检测所有人脸，识别结果包含 `detected_faces` 数组。

### Q6: 如何集成到现有系统？

**A**:
- **RESTful API**: 直接调用 `/recognize` 和 `/persons` 接口
- **SDK 封装**: 参考 `test_recognize.py` 示例代码
- **Webhook**: 在 `router/faces.py` 中添加识别完成回调

### Q7: InsightFace 和 dlib 如何选择？

**A**:
- **InsightFace + FastDeploy**: 推荐用于生产环境，检测精度更高，GPU 加速，处理速度快
- **dlib + FastDeploy**: 适用于资源受限环境，CPU 友好，模型较小，稳定可靠
- **混合模式**: 系统自动选择最优引擎，InsightFace 失败时回退到 dlib
- **特征提取**: 两种路径都使用 FastDeploy ArcFace 模型，确保特征向量一致性

---

## 测试脚本

### 识别功能测试 (`test_recognize.py`)

```bash
cd /root/workspace/FaceRecAPI_DEV/app
python test_recognize.py
```

**功能**:
- 上传测试图片进行识别
- 验证 API 响应格式
- 打印识别结果和相似度

### 压力测试 (`stress_test.py`)

```bash
python stress_test.py
```

**功能**:
- 并发请求测试
- 性能指标统计 (QPS, 响应时间)
- 内存占用监控

---

## 许可证

本项目为内部开发项目，版权所有。

---

## 联系方式

如有问题或建议，请联系开发团队。

---

**文档版本**: v2.0 (InsightFace 版本)
**更新时间**: 2025-01-18
**项目路径**: `/root/workspace/FaceRecAPI_DEV`
**主要变更**:
- ✅ 从 dlib 迁移到 InsightFace 0.7.3 (检测对齐)
- ✅ 保留 FastDeploy ArcFace (特征提取)
- ✅ 新增 Redis 缓存系统 (Pickle 序列化优化)
- ✅ 三重混合引擎 (InsightFace + FastDeploy + dlib 智能回退)
- ✅ GPU 加速优化 + 版本兼容性调整
