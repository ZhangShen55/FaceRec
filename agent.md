# FaceRecAPI_DEV 项目文档

## 项目概述

**FaceRecAPI_DEV** 是基于深度学习的高性能人脸比对 API 系统，提供完整的人脸检测、关键点对齐、特征提取、相似度匹配、人物库管理与运维监控能力。

### 核心特性

- **N:N 多人脸识别**：单图同时检测多张人脸，逐脸进行候选/全局双阶段匹配
- **多帧识别融合**：`/recognize/batch` 支持视频/连拍多帧聚合，按 `number` 去重取最高相似度
- **三段式引擎**：InsightFace（检测+对齐）+ FastDeploy ArcFace（512 维特征）+ dlib（智能回退）
- **GPU 优先 + 智能降级**：GPU 失败自动降到 CPU；InsightFace 不可用自动降到 dlib，确保系统不中断
- **MongoDB + Redis**：MongoDB 存特征/人物，Redis 用 Pickle 缓存全量特征，相似度计算无需查库
- **启动预加载**：进程启动时把 InsightFace 子进程模型 + 主进程 ArcFace 模型预热到 GPU，首请求耗时 ↓ 70-80%
- **ROI 多边形掩膜**：`points` 参数支持任意多边形 ROI，仅识别区域内人脸
- **统一响应规范**：HTTP 永远 200，业务态由 `statusCode` 字段细分
- **API 统计中间件**：所有请求落 MongoDB 明细 + 按小时聚合，TTL 索引自动清理
- **Web 管理 UI**：Bootstrap 5 + Vanilla JS + HTTP Basic 鉴权，支持在线识别与人物库管理

---

## 技术架构

### 技术栈

| 层级 | 技术 | 版本/说明 |
|------|------|---------|
| **Web 框架** | FastAPI + Uvicorn / Gunicorn | 异步 Web 服务器 |
| **数据库** | MongoDB（Motor 异步驱动） | NoSQL，存人物 + 统计数据 |
| **缓存** | Redis（`redis[hiredis]>=5.0.0`） | Pickle 序列化全量特征 |
| **检测+对齐（主）** | InsightFace 0.7.3（buffalo_l） | GPU 加速 SCRFD + 5 点 norm_crop |
| **特征提取（主）** | FastDeploy-GPU 1.0.7（ArcFace R100） | 512 维 L2 归一化向量 |
| **检测+对齐（备）** | dlib | CPU，68 点关键点 → 5 点仿射对齐 |
| **推理后端** | onnxruntime-gpu 1.16.3 | 兼容 CUDA 11.6/11.8 |
| **图像处理** | OpenCV + NumPy 1.26.4 | NumPy 锁定在 1.x 以兼容 ORT |
| **前端** | Bootstrap 5 + Vanilla JS | 响应式 Web UI |
| **配置管理** | TOML（tomli） | 单文件配置，强类型 Pydantic 模型 |

### 模型清单（详见 `app/ai_models/README.md`）

- **InsightFace `buffalo_l`**：`det_10g.onnx` / `w600k_r50.onnx` / `2d106det.onnx` / `1k3d68.onnx` / `genderage.onnx`
- **FastDeploy ArcFace**：`ms1mv3_arcface_r100.onnx`（**全系统统一的 512 维特征向量来源**）
- **dlib**：`shape_predictor_68_face_landmarks.dat`（备用引擎）

> 当前实现里特征向量**始终由 FastDeploy ArcFace 输出**，无论检测路径是 InsightFace 还是 dlib，从而保证特征空间一致、可互比对。

### 系统架构图

```
┌──────────────────────────────────────────────────────────────┐
│                客户端（浏览器 / API 调用方 / SDK）            │
└──────────────────────────────┬───────────────────────────────┘
                               │ HTTP/HTTPS
┌──────────────────────────────▼───────────────────────────────┐
│  FastAPI 应用层 (main.py + lifespan)                         │
│  ┌──────────────┬──────────────┬──────────────┬───────────┐ │
│  │ /recognize   │ /persons     │ /ops         │ Web UI /  │ │
│  │ /recognize   │ /persons     │ health /     │ static    │ │
│  │  /batch      │  /batch      │ metrics /    │ media     │ │
│  │              │ /search      │ stats/*      │           │ │
│  │              │ /delete      │              │           │ │
│  └──────────────┴──────────────┴──────────────┴───────────┘ │
│  中间件：APIStatsMiddleware + add_request_id                 │
└──────────────────────────────┬───────────────────────────────┘
                               │
┌──────────────────────────────▼───────────────────────────────┐
│  业务服务层 (services/)                                       │
│  ┌─────────────┐ ┌──────────────┐ ┌───────────────────────┐ │
│  │ person.py   │ │ cache_service │ │ ops_stats / face_svc │ │
│  │ CRUD        │ │ Redis Pickle  │ │ 聚合统计 / 识别封装   │ │
│  └─────────────┘ └──────────────┘ └───────────────────────┘ │
└──────────┬───────────────────────────┬────────────────────────┘
           │                           │
┌──────────▼─────────────┐  ┌──────────▼─────────────┐  ┌──────────────┐
│ AI 引擎 (core/         │  │ MongoDB                │  │ Redis        │
│ ai_engine.py)          │  │ ┌────────────────────┐ │  │ ┌──────────┐ │
│ ┌────────────────────┐ │  │ │ persons            │ │  │ │ embeddings│ │
│ │ InsightFace        │ │  │ │  - name/number     │ │  │ │ (Pickle  │ │
│ │ buffalo_l (GPU)    │ │  │ │  - photo_path/bbox │ │  │ │  全量)   │ │
│ │ 检测 + 5pt 对齐    │ │  │ │  - embedding(Bin)  │ │  │ └──────────┘ │
│ └────────────────────┘ │  │ └────────────────────┘ │  │ key: face:   │
│ ┌────────────────────┐ │  │ ┌────────────────────┐ │  │  embeddings: │
│ │ FastDeploy ArcFace │ │  │ │ api_call_logs      │ │  │  all         │
│ │ R100 (GPU)         │ │  │ │ (TTL retention_d)  │ │  └──────────────┘
│ │ 512d 特征向量      │ │  │ └────────────────────┘ │
│ └────────────────────┘ │  │ ┌────────────────────┐ │
│ ┌────────────────────┐ │  │ │ api_stats_hourly   │ │
│ │ dlib (CPU 回退)    │ │  │ │ (TTL hourly_d)     │ │
│ └────────────────────┘ │  │ └────────────────────┘ │
│ ┌────────────────────┐ │  └────────────────────────┘
│ │ 双阶段匹配 +       │ │
│ │ ProcessPoolExecutor│ │
│ └────────────────────┘ │
└────────────────────────┘
```

---

## 目录结构

```
FaceRecAPI_DEV/
└── app/                                  # 主应用包
    ├── main.py                           # FastAPI 入口（lifespan + 模型预加载）
    ├── config.toml                       # 项目配置（主源） ⚙️
    ├── requirements.txt                  # Python 依赖
    ├── agent.md                          # 本文档 📄
    ├── README.md                         # 简版说明
    ├── .gitignore
    │
    ├── core/                             # 核心基础设施
    │   ├── ai_engine.py                  # AI 引擎核心（InsightFace + dlib + FastDeploy）
    │   ├── config.py                     # TOML 配置加载（Pydantic）
    │   ├── database.py                   # MongoDB Motor 客户端
    │   ├── redis_client.py               # Redis 客户端（异步 hiredis）
    │   ├── logger.py                     # 日志 + request_id 上下文
    │   ├── exceptions.py                 # 自定义异常
    │   └── constants.py                  # 常量定义
    │
    ├── models/                           # Pydantic 数据模型
    │   ├── api_response.py               # 统一响应 ApiResponse + StatusCode 枚举
    │   ├── schemas.py                    # 通用基础模型
    │   ├── request/                      # 请求模型
    │   │   ├── face_interface_req.py     # PersonRecognizeRequest / BatchRecognizeRequest / Point
    │   │   ├── person_interface_req.py   # PersonFeatureRequest / Search/Delete...
    │   │   └── ops_interface_req.py
    │   └── response/
    │       ├── face_interface_rep.py     # BBox / MatchItem / RecognizeResp / FrameInfo / BatchRecognizeResp
    │       ├── person_interface_rep.py
    │       └── ops_interface_rep.py
    │
    ├── router/                           # API 路由
    │   ├── faces.py                      # /recognize 与 /recognize/batch
    │   ├── faces_backup.py               # 旧版备份（不参与挂载）
    │   ├── persons.py                    # /persons CRUD + 批量 + 搜索 + 删除
    │   ├── ops.py                        # /ops/health /metrics /stats/*
    │   └── web.py                        # Web UI（HTTP Basic）
    │
    ├── services/                         # 业务逻辑层
    │   ├── person.py                     # MongoDB 人物 CRUD
    │   ├── cache_service.py              # Redis 缓存（Pickle v2，自动从 v1 JSON 迁移）
    │   ├── face_service.py               # 识别流程封装（保留模块）
    │   └── ops_stats.py                  # API 统计聚合查询
    │
    ├── middleware/                       # Starlette/FastAPI 中间件
    │   ├── __init__.py                   # 导出 APIStatsMiddleware
    │   └── api_stats_middleware.py       # 请求拦截 + 写入 api_call_logs / api_stats_hourly
    │
    ├── utils/
    │   ├── image_loader.py               # base64_to_mat / 像素校验 / apply_polygon_roi
    │   └── utils_mongo.py                # MongoDB 文档转 Pydantic 工具
    │
    ├── ai_models/                        # AI 模型文件（详见 ai_models/README.md）
    │   ├── README.md
    │   ├── shape_predictor_68_face_landmarks.dat
    │   ├── ms1mv3_arcface_r100.onnx
    │   └── models/buffalo_l/{det_10g,w600k_r50,2d106det,1k3d68,genderage}.onnx
    │
    ├── static/                           # Web 前端资源
    │   ├── templates/index.html
    │   ├── js/app.js
    │   └── css/
    │
    ├── media/                            # 已裁剪人脸照片
    │   └── person_photos/
    │
    ├── logs/                             # 运行日志（uvicorn / facerecapi）
    │
    ├── docker/                           # 容器化部署
    │   ├── Dockerfile                    # nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04
    │   ├── docker-compose.yml
    │   └── nginx.conf
    │
    ├── docs/                             # 详细 API 文档
    │   ├── PERSONS_API.md
    │   ├── RECOGNIZE_API.md
    │   ├── OPS_API.md
    │   ├── PERSONS_DELETE_API_OPTIMIZATION.md
    │   ├── PROJECT_STRUCTURE_OPTIMIZATION.md
    │   └── OPTIMIZATION_SUMMARY.md
    │
    ├── tests/                            # 测试与压测
    │   ├── test_recognize.py             # 识别接口烟雾测试
    │   ├── stress_test_n.py              # 并发压测（N:N 多人脸）
    │   ├── integration/
    │   ├── unit/
    │   └── 测试人员.jpg / 常泽宇.png      # 测试图片
    │
    ├── scripts/                          # 运维脚本（预留）
    └── tmp/                              # 临时备份/草稿
```

---

## 核心功能

### 1. 单图 N:N 多人脸识别（`POST /recognize`）

```
输入图片 (Base64) + 可选 targets + 可选 ROI 多边形
       │
       ▼
┌──────────────────────────────┐
│ 1. base64_to_mat             │ 解析 + 像素总数校验（80x80 ~ max）
└──────────────┬───────────────┘
               │
               ▼ (可选)
┌──────────────────────────────┐
│ 2. apply_polygon_roi(points) │ 多边形掩膜，区域外置黑
└──────────────┬───────────────┘
               │
               ▼
┌──────────────────────────────┐
│ 3. detect_and_extract_       │ ProcessPool → 子进程
│    all_faces                 │   - InsightFace.get(image) (主)
│                              │   - dlib + 68pt → 5pt 仿射 (备)
│                              │ 返回 List[(aligned 112x112, bbox, tip)]
└──────────────┬───────────────┘
               │
               ▼
┌──────────────────────────────┐
│ 4. 过滤 rec_min_face_hw      │ shape < 50px 的人脸丢弃
└──────────────┬───────────────┘
               │
               ▼
┌──────────────────────────────┐
│ 5. cache_service.get_all_    │ Redis Pickle 全量特征（毫秒级）
│    embeddings                │
└──────────────┬───────────────┘
               │
               ▼
┌──────────────────────────────┐
│ 6. 对每张人脸：              │
│    - get_embedding (GPU)      │
│    - 阶段1: targets matches   │ 阈值 = candidate_threshold (0.2)
│    - 阶段2: 全库 matches     │ 阈值 = threshold (0.4)
└──────────────┬───────────────┘
               │
               ▼
┌──────────────────────────────┐
│ 7. 跨脸合并 + number 去重    │ 按相似度降序，最多 top 3
└──────────────┬───────────────┘
               │
               ▼
返回 ApiResponse(statusCode=200, match=[MatchItem(bbox, name, number, similarity, is_target), ...])
```

### 2. 多帧识别融合（`POST /recognize/batch`）

- 每帧独立走"检测 → 特征 → 全库 + targets 匹配"
- 跨帧按 `number` 聚合：取最高相似度，记录出现次数 / `is_target` OR 合并
- 返回每帧明细 `frames[]` + 聚合 top 3 `match[]`，方便前端定位失败帧

### 3. 双阶段匹配策略

| 阶段 | 候选集合 | 阈值 | 设计意图 |
|------|---------|------|---------|
| 阶段 1 | `targets` 指定的 `number` 列表 | `candidate_threshold = 0.2` | 宽松，提高目标人物召回率 |
| 阶段 2 | 全库 | `threshold = 0.4` | 严格，控制误识别 |

实现见 `router/faces.py`：

```python
# 阶段 1：targets 优先匹配（候选阈值）
if target_docs:
    target_results = ai_engine.find_top_matches(
        emb_q, target_docs, top_k=3, min_threshold=CANDIDATE_THRESHOLD
    )
    for sim, doc in target_results:
        face_matches.append((sim, doc, bbox, True))  # is_target=True

# 阶段 2：全局匹配（严格阈值）
global_results = ai_engine.find_top_matches(
    emb_q, all_docs, top_k=3, min_threshold=threshold
)
```

### 4. 引擎选择与智能回退（`core/ai_engine.py`）

- `_init_dlib_worker()` 是 `ProcessPoolExecutor` 的 `initializer`，在每个子进程启动时根据 `face_detection.detector` 配置加载模型：
  - `insightface` → 尝试加载 `buffalo_l`（GPU 失败自动降级到 CPU；CPU 也失败再降级到 dlib）
  - `dlib` → 直接加载 dlib 检测器 + 68 点预测器
- 主进程不导入 InsightFace，避免 fork 时继承 GPU 句柄。
- ArcFace 模型在主进程**延迟单例加载**（`_get_embedding_model` + 双重检查锁），通过 `asyncio.to_thread` 调用，避免阻塞事件循环。

### 5. GPU 重映射机制

`config.toml` 中 `[face_detection.insightface].gpu_id` 指定**逻辑 GPU**：

- 若 `gpu_id != 0`，`main.py` 在导入 `ai_engine` 之前设置 `CUDA_VISIBLE_DEVICES = gpu_id`
- 子进程内部使用映射后的 `actual_gpu_id = 0`，避免在 GPU 0 上残留资源
- `[gpu].gpu_id` 用于 ArcFace 模型，逻辑同上

### 6. 启动预加载

`main.py::lifespan` 启动顺序：

1. MongoDB ping
2. Redis ping
3. `cache_service.reload_all_embeddings()` 全量入 Redis（仅当 `redis.cache.refresh_on_startup=true`）
4. 创建 dlib `ProcessPoolExecutor`（`max_workers = thread.max_workers`）+ initializer 触发模型加载
5. **预热**：向进程池提交一个 dummy 任务触发子进程内 InsightFace 加载；主进程同步加载 ArcFace
6. yield → 服务对外可用
7. 关闭：Redis close + 进程池 shutdown

效果：首次识别请求从 10-15s 降到 2-3s（提升 70-80%）。

### 7. 特征向量管理

| 维度 | 值 |
|------|----|
| 维度 | 512 |
| dtype | float32 |
| 归一化 | L2，`||emb|| = 1` |
| 相似度 | 余弦 = `np.dot(db_vecs, emb_q)` 批量计算 |
| MongoDB 存储 | `Binary(emb.tobytes())` |
| Redis 存储 | Pickle（HIGHEST_PROTOCOL），保留 numpy array 原生格式 |

---

## API 接口文档

> **统一规范**：所有接口 HTTP 状态码恒为 **200**，业务态由响应体的 `statusCode` 字段（驼峰）区分。

### 状态码总览（`models/api_response.py`）

| statusCode | 含义 |
|------------|------|
| 200 | 成功 |
| 201 | 未检测到人脸 (`NO_FACE_DETECTED`) |
| 202 | 人脸像素过小 (`FACE_TOO_SMALL`) |
| 207 | 批量部分成功 (`PARTIAL_SUCCESS`) |
| 251 | 数据库为空 (`DB_EMPTY`) |
| 252 | 未匹配到对象（低于阈值）(`NO_MATCH_FOUND`) |
| 400 | 请求参数错误 |
| 401 | base64 解码失败 |
| 402 | 图片格式错误（cv2 解析失败）|
| 403 | 未接收到有效图片数据 |
| 404 | 资源未找到 |
| 422 | 无法处理的实体（保留） |
| 500 | 服务器内部错误 |
| 501 | 人脸检测服务错误 |
| 502 | 特征提取失败 |
| 503 | 文件保存失败 |

### 路由清单

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/recognize` | 单图 N:N 多人脸识别 |
| POST | `/recognize/batch` | 多帧融合识别 |
| POST | `/persons` | 新增或按 `number` 更新单个人物 |
| POST | `/persons/batch` | 批量新增/更新 |
| GET | `/persons` | 分页查询人物列表 |
| POST | `/persons/search` | 按 `name`(模糊) / `number`(精确) 查询 |
| DELETE | `/persons/delete` | 通用删除（`name`/`number`/`id`，三选一） |
| GET | `/ops/health` | 健康检查（DB + 磁盘） |
| GET | `/ops/metrics` | 系统指标（CPU/内存/磁盘 + 应用指标） |
| GET | `/ops/stats/api-calls` | API 调用明细查询 |
| GET | `/ops/stats/hourly` | 按小时聚合统计 |
| GET | `/ops/stats/summary` | 汇总统计（成功率、Top N 接口、按小时分布） |
| GET | `/` | Web 管理 UI（HTTP Basic 鉴权） |

### 1. `POST /recognize`

**请求体**：

```json
{
  "photo": "data:image/jpeg;base64,/9j/4AAQSkZJRg...",
  "targets": ["T001", "T002"],
  "threshold": 0.4,
  "points": [
    {"x": 0,    "y": 0},
    {"x": 1920, "y": 0},
    {"x": 1920, "y": 540},
    {"x": 0,    "y": 540}
  ]
}
```

- `photo`：必填，`data:image/...;base64,` 前缀的 Base64 字符串
- `targets`：可选，候选 `number` 列表，触发阶段 1 候选匹配
- `threshold`：可选，覆盖全局阈值（默认 `face.threshold = 0.4`）
- `points`：可选，多边形 ROI 顶点，**至少 3 个，按顺时针排列**，超出图像边界的坐标自动 clip

**成功响应**：

```json
{
  "statusCode": 200,
  "message": "识别成功",
  "data": {
    "hasFace": true,
    "threshold": 0.4,
    "match": [
      {
        "bbox": {"x": 100, "y": 120, "w": 150, "h": 150},
        "id": "507f1f77bcf86cd799439011",
        "name": "张三",
        "number": "T001",
        "similarity": "87.45%",
        "is_target": true
      }
    ],
    "message": "匹配成功，≥阈值40.00%有1位，最相似的是张三_T001"
  }
}
```

**典型错误响应**：

| statusCode | 场景 |
|------------|------|
| 401/402/403 | 图片解析失败 / 格式错误 / 未接收到有效数据 |
| 201 | 未检测到人脸 |
| 202 | 检测到的人脸全部过小 |
| 251 | 数据库为空（仍返回 `bboxs` 数组） |
| 252 | 检测到人脸但相似度全部低于阈值 |
| 501 | 检测服务异常（进程池/InsightFace 报错） |

### 2. `POST /recognize/batch`

**请求体**：

```json
{
  "photos": ["data:image/jpeg;base64,...", "data:image/jpeg;base64,..."],
  "targets": ["T001"],
  "threshold": 0.4
}
```

**响应**：

```json
{
  "statusCode": 200,
  "message": "批量识别成功",
  "data": {
    "total_frames": 5,
    "valid_frames": 4,
    "threshold": 0.4,
    "frames": [
      {"index": 0, "hasFace": true,  "bbox": {"x":100,"y":120,"w":150,"h":150}, "error": null},
      {"index": 1, "hasFace": false, "bbox": null, "error": "未检测到人脸"}
    ],
    "match": [
      {"id": "...", "name": "张三", "number": "T001", "similarity": "89.12%", "is_target": true}
    ],
    "message": "识别成功，使用4帧有效图片，找到1位候选人，targets命中1位，最相似的是张三_T001（出现3次）"
  }
}
```

### 3. `POST /persons`

```json
{ "name": "张三", "number": "T001", "photo": "data:image/jpeg;base64,..." }
```

- 按 `number` 去重：存在则**更新**，不存在则**创建**（`update_or_create_person`）
- 副作用：保存裁剪头像到 `app/media/person_photos/<name>_<number>_<uuid8>.jpg`
- 触发 Redis 缓存全量刷新（如启用 `refresh_on_update`）

### 4. `POST /persons/batch`

```json
{
  "persons": [
    { "name": "张三", "number": "T001", "photo": "data:image/jpeg;base64,..." },
    { "name": "李四", "number": "T002", "photo": "data:image/jpeg;base64,..." }
  ]
}
```

- 全部成功 → `statusCode = 200`
- 部分失败 → `statusCode = 207`，`data.failed_numbers` / `failed_details` 给出失败编号与原因
- 全部失败 → `statusCode = 400`

### 5. `GET /persons`

Query：`skip`（默认 0）、`limit`（默认 100）。返回 `data.persons[]`，**不包含特征向量**。

### 6. `POST /persons/search`

```json
{ "name": "张", "number": "T001" }   // 至少传一个，组合时 name 模糊 AND number 精确
```

返回最多 20 条，**不包含特征向量**。

### 7. `DELETE /persons/delete`

```json
{ "name": "张三" }    // 按姓名模糊（可能命中多个）
{ "number": "T001" }  // 按编号精确
{ "id": "65abc..." }  // 按 MongoDB _id 精确
```

优先级：`name` > `number` > `id`。删除后触发 Redis 缓存全量刷新。

### 8. `/ops/*` 运维接口

- `GET /ops/health`：返回 `{status: healthy/degraded, components: {database, storage}}`
- `GET /ops/metrics`：CPU/内存/磁盘 + 应用指标（人物总数等）
- `GET /ops/stats/api-calls?start_date=&end_date=&endpoint=&method=&limit=&offset=`：明细查询
- `GET /ops/stats/hourly?...`：按小时聚合
- `GET /ops/stats/summary?start_date=&end_date=`：汇总（默认最近 7 天）

### 9. Web UI

- `GET /`：HTTP Basic 鉴权（`frontlogin.username/password`）
- 提供：在线识别、人物列表、批量录入、删除等

---

## 配置文件详解

### config.toml 全字段

```toml
# ====== MongoDB ======
[db]
username = "root"
password = "root"
host = "10.80.5.25"
port = "27017"
database = "facerecapi"
auth_source = "admin"
limit = 5000                             # 单次查询最大条数（用于 get_embeddings_for_match）

# ====== 人脸识别参数 ======
[face]
threshold = 0.4                          # 全局严格阈值
candidate_threshold = 0.2                # targets 候选阶段阈值
rec_min_face_hw = 50                     # 识别阶段对齐后最小人脸尺寸 (px)

# ====== 检测引擎选择 ======
[face_detection]
detector = "insightface"                 # "insightface" | "dlib"

[face_detection.insightface]
model_name = "buffalo_l"                 # buffalo_l (高精度) | buffalo_s (轻量)
model_path = "/root/workspace/FaceRecAPI_DEV/app/ai_models"  # 指向 ai_models/，内部会拼 models/<name>/
device = "gpu"                           # "gpu" | "cpu"
gpu_id = 1                               # 逻辑 GPU 编号（启动时设置 CUDA_VISIBLE_DEVICES）
det_size = 320                           # 检测尺寸：320 / 640 / 1280
det_thresh = 0.75                        # InsightFace 置信度阈值（低于此值直接丢弃）

# ====== ArcFace GPU 配置 ======
[gpu]
gpu_id = 1                               # FastDeploy ArcFace 使用的 GPU 编号

# ====== 进程池（dlib worker / InsightFace 子进程数）======
[threading]
max_workers = 1                          # GPU InsightFace 强烈建议 1，避免显存竞争；纯 dlib CPU 可设为 2

# ====== Web UI 登录 ======
[frontlogin]
username = "admin"
password = "admin"

# ====== 图像验证 ======
[image]
max_feature_image_width_px = 9999
max_feature_image_height_px = 9999
min_feature_image_width_px = 80          # ⚠️ 影响 base64_to_mat 校验（最低分辨率 80x80）
min_feature_image_height_px = 80
max_feature_image_size_m = 10            # 文件大小上限 (MB)
max_face_hw = 999
min_face_hw = 50

# ====== 日志 ======
[logger]
level = "INFO"                           # DEBUG / INFO / WARNING / ERROR
log_path = "./app/logs/facere_service.log"

# ====== API 调用统计 ======
[stats]
retention_days = 7                       # api_call_logs 明细保留天数（TTL）
hourly_retention_days = 30               # api_stats_hourly 聚合保留天数（TTL）

# ====== Redis ======
[redis]
host = "localhost"
port = 6379
password = ""
db = 0
max_connections = 500
socket_timeout = 30
socket_connect_timeout = 10
decode_responses = false                 # 必须 false 以支持 Pickle 二进制读写

[redis.cache]
embeddings_ttl = 0                       # 0 = 永不过期（手动刷新）；>0 秒数过期
enable_embedding_cache = true            # 总开关
refresh_on_startup = true                # 启动时全量刷新
refresh_interval_days = 7                # 定时刷新周期（保留字段，由外部调度执行）
refresh_on_update = true                 # 增/改/删时同步刷新
```

### 配置调优建议

| 配置 | 默认 | 建议 |
|------|------|------|
| `face.threshold` | 0.4 | 严要求场景 ≥ 0.45；高召回场景 0.35-0.40 |
| `face.candidate_threshold` | 0.2 | 一般保持低值，仅用于召回 targets |
| `face_detection.insightface.det_size` | 320 | 小人脸/远景调高到 640；高吞吐近景可保 320 |
| `face_detection.insightface.det_thresh` | 0.75 | 误检多调高、漏检多调低（常用 0.5–0.8） |
| `threading.max_workers` | 1 | InsightFace GPU 模式必须为 1；纯 dlib CPU 可 = 物理核数 50%-75% |
| `image.min_feature_image_width/height_px` | 80 | 影响 `base64_to_mat` 像素总数下限（80×80=6400） |
| `redis.cache.embeddings_ttl` | 0 | 人物库变化频繁可改为 300-3600；变化少建议永不过期 |
| `db.limit` | 5000 | 万级人物库可调至 50000；十万级建议接 Faiss/Milvus |

---

## 数据库设计

### 1. `persons` 集合（人物库）

```javascript
{
  "_id": ObjectId("65abc..."),
  "name": "张三",                      // 可重复
  "number": "T001",                    // 应用层去重（update_or_create_person）
  "photo_path": "/media/person_photos/张三_T001_a1b2c3d4.jpg",
  "bbox": "100,120,150,150",           // 录入时检测到的 bbox: x,y,w,h
  "embedding": BinData(...),           // 512 维 float32 L2 归一化向量
  "tip": "人脸特征像素正常，可以使用"   // 录入时质量提示
}
```

**索引建议**（部署时手动创建，代码未自动建）：

```javascript
db.persons.createIndex({ "number": 1 }, { unique: true })
db.persons.createIndex({ "name": 1 })
```

### 2. `api_call_logs` 集合（请求明细）

由 `APIStatsMiddleware` 写入，每个非静态请求一条。**自动 TTL 清理**（`stats.retention_days` 天）。

```javascript
{
  "request_id": "uuid",
  "timestamp": ISODate("..."),    // TTL 索引字段
  "date": "2026-01-30",
  "hour": 10,
  "method": "POST",
  "path": "/recognize",
  "status_code": 200,             // 注意：HTTP 状态码，而非业务 statusCode
  "duration_ms": 235.42,
  "client_ip": "10.0.0.1",
  "success": true,
  "error_message": null
}
```

### 3. `api_stats_hourly` 集合（按小时聚合）

按 `(date, hour, endpoint, method)` upsert 聚合，**自动 TTL 清理**（`stats.hourly_retention_days` 天）。

```javascript
{
  "date": "2026-01-30",
  "hour": 10,
  "endpoint": "/recognize",
  "method": "POST",
  "total_requests": 150,
  "success_count": 145,
  "error_count": 5,
  "response_times": [235.42, 198.13, ...],   // 仅保留最近 1000 条
  "min_response_time_ms": 120.5,
  "max_response_time_ms": 580.2,
  "last_updated": ISODate("...")              // TTL 索引字段
}
```

中间件首次启动时通过 `_ensure_indexes` 自动创建 TTL 索引及查询优化索引。

---

## 部署指南

### 环境要求

**硬件**：
- CPU：4 核+（推荐 8 核）
- 内存：16 GB+（推荐 32 GB，InsightFace + ArcFace 同驻 GPU/内存）
- GPU：NVIDIA GPU，显存 ≥ 8 GB（InsightFace 4-6 GB + ArcFace 2-3 GB），推荐 RTX 3090/4090/A100
- 磁盘：≥ 50 GB（模型 ~700 MB + 头像 + 日志）

**软件**：
- Python 3.10（与 Dockerfile 一致；3.8 也可运行但未测试）
- MongoDB 4.4+
- Redis 6.0+
- CUDA 11.6 / 11.8（与 `onnxruntime-gpu==1.16.3` 兼容）
- OS：Ubuntu 22.04 / CentOS 7+

### 1. 安装依赖

```bash
cd /root/workspace/FaceRecAPI_DEV/app
pip install -r requirements.txt
```

`requirements.txt` 关键依赖：

```text
fastapi
uvicorn[standard]
gunicorn
motor
pymongo
redis[hiredis]>=5.0.0
numpy==1.26.4
opencv-python
dlib
insightface==0.7.3
onnxruntime-gpu==1.16.3
fastdeploy-gpu-python==1.0.7
psutil
tomli
```

### 2. 准备模型文件

详见 [`app/ai_models/README.md`](./ai_models/README.md)。最终目录：

```
app/ai_models/
├── shape_predictor_68_face_landmarks.dat
├── ms1mv3_arcface_r100.onnx
└── models/buffalo_l/{det_10g,w600k_r50,2d106det,1k3d68,genderage}.onnx
```

### 3. 配置数据库与 Redis

```bash
# MongoDB
systemctl start mongod
mongo
> use facerecapi
> db.createUser({user:"root", pwd:"root", roles:[{role:"readWrite", db:"facerecapi"}]})
> db.persons.createIndex({number:1}, {unique:true})

# Redis
systemctl start redis
redis-cli ping   # 期望 PONG
```

### 4. 修改 `config.toml`

按实际环境调整 `[db]` / `[redis]` / `[face_detection.insightface].gpu_id` / `[gpu].gpu_id` / `[frontlogin]`。

### 5. 启动服务

**开发模式（单进程 + 热重载）**：

```bash
cd /root/workspace/FaceRecAPI_DEV
PYTHONPATH=/root/workspace/FaceRecAPI_DEV OMP_NUM_THREADS=1 \
  uvicorn app.main:app --host 0.0.0.0 --port 8003 --reload
```

**生产模式（推荐 Uvicorn 单 worker）**：

```bash
cd /root/workspace/FaceRecAPI_DEV/app
nohup env PYTHONPATH=/root/workspace/FaceRecAPI_DEV OMP_NUM_THREADS=1 \
  uvicorn app.main:app --host 0.0.0.0 --port 8003 --workers 1 \
  > /root/workspace/FaceRecAPI_DEV/app/logs/facerec_server_uvicorn.log 2>&1 &
echo $! > /root/workspace/FaceRecAPI_DEV/app/logs/facerec_server_uvicorn.pid
```

> **必须 `--workers 1`**：InsightFace + ArcFace 都驻留 GPU，多 worker 会导致显存翻倍且竞争。

**Gunicorn 替代**：

```bash
PYTHONPATH=/root/workspace/FaceRecAPI_DEV OMP_NUM_THREADS=1 \
  gunicorn -w 1 -k uvicorn.workers.UvicornWorker \
  --bind 0.0.0.0:8003 --timeout 120 app.main:app
```

**预期启动日志**：

```
✅ MongoDB ping ok
✅ Redis 连接成功
✅ 启动时已加载 XXX 个人员特征到 Redis
正在初始化 Dlib 进程池，工作线程数: 1...
✅ Dlib 进程池初始化完成
🔄 预加载 AI 模型到 GPU...
🔄 预加载 InsightFace 检测模型...
✅ InsightFace 已预加载
🔄 预加载 Embedding 模型 (ArcFace)...
✅ Embedding 模型已预加载到 GPU
✅ AI 模型预加载完成
Uvicorn running on http://0.0.0.0:8003
```

### 6. 容器化部署

参考 `app/docker/`：

```bash
cd /root/workspace/FaceRecAPI_DEV/app
docker compose -f docker/docker-compose.yml up -d --build
```

`docker-compose.yml` 默认：
- 镜像：`nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04`
- 端口：`8000:8000`（如需对齐本地 8003 请修改）
- 挂载：`config.toml` / `logs/` / `ai_models/`（只读）
- GPU：`driver: nvidia, count: 1, capabilities: [gpu]`

### 7. systemd 托管

```ini
# /etc/systemd/system/facerecapi.service
[Unit]
Description=FaceRecAPI Service
After=network.target mongod.service redis.service

[Service]
Type=simple
User=root
WorkingDirectory=/root/workspace/FaceRecAPI_DEV
Environment="PYTHONPATH=/root/workspace/FaceRecAPI_DEV"
Environment="OMP_NUM_THREADS=1"
ExecStart=/usr/local/bin/uvicorn app.main:app --host 0.0.0.0 --port 8003 --workers 1
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl daemon-reload && sudo systemctl enable --now facerecapi
```

### 8. 健康验证

```bash
curl http://localhost:8003/ops/health     # 期望 status: healthy
curl http://localhost:8003/ops/metrics    # 系统/应用指标
# Web UI: http://<host>:8003   账号 admin/admin（按配置）
```

---

## 性能指标

| 指标 | 数值 | 说明 |
|------|------|------|
| 启动耗时 | 15-30 s | 包含 MongoDB/Redis/特征缓存 + 模型预加载 |
| 首次识别 | 2-3 s | 预加载后；未预加载时 10-15 s |
| 后续识别 | 200-500 ms | 单图单脸常规耗时（GPU） |
| 多人脸识别 | +50-150 ms/脸 | 按检测到的人脸数量线性增长 |
| GPU 显存 | 6-9 GB | InsightFace ~4-6 + ArcFace ~2-3 |
| Redis 全量加载 | < 100 ms | 1 万人物量级 |

GPU 兼容性：RTX 3090/4090/A100 ✅；T4(16GB) ⚠️ 紧张；显存 < 8 GB 建议切 dlib 或 `buffalo_s`。

---

## 开发指南

### 代码分层与依赖方向

```
router/  →  services/  →  core/(database, ai_engine, redis_client)
   ↓                        ↑
models/  ←  utils/  ───────┘
middleware/ ⊂ APIStatsMiddleware → core/database
```

### 添加新 API 端点

1. 在 `models/request/` 定义请求模型（推荐用 `field_validator` 做必填校验）
2. 在 `models/response/` 定义响应模型
3. 在 `services/` 实现业务逻辑
4. 在 `router/` 创建路由模块，统一返回 `ApiResponse.success(...)` / `ApiResponse.error(status_code=StatusCode.XXX, ...)`
5. 在 `main.py` 用 `app.include_router(...)` 注册

### 调试

```bash
# 实时日志
tail -f /root/workspace/FaceRecAPI_DEV/app/logs/facerec_server_uvicorn.log

# 开 DEBUG 日志
# config.toml -> [logger].level = "DEBUG"

# 识别冒烟测试
cd /root/workspace/FaceRecAPI_DEV
python -m app.tests.test_recognize

# 并发压测（N:N 多人脸）
python -m app.tests.stress_test_n
```

### 性能优化

1. **数据库**：建立 `persons.number` 唯一索引；查询时投影 `{embedding:1, name:1, number:1, photo_path:1}` 减少网络
2. **Redis**：保持 `embeddings_ttl=0` 永不过期，避免冷启动；变更后由 `cache_service.update_person_cache` 触发刷新
3. **InsightFace**：`det_size=320` 适配近景；远景小脸 `det_size=640` + `det_thresh=0.5`
4. **进程池**：GPU InsightFace 模式 `max_workers=1`；纯 dlib CPU 模式可调高
5. **大库扩展**：> 10 万人物量请引入 Faiss/Milvus 向量索引层，替换 `find_top_matches` 的全量点积

---

## 常见问题（FAQ）

### Q1：识别准确率不理想？

- 提高 `face.threshold` 减少误识别（如 0.45-0.5）
- 录入照片质量很关键：正脸、光照充足、像素 ≥ 200×200
- 检查 `face_detection.insightface.det_thresh`：太低会引入噪声脸

### Q2：GPU 显存不足？

- 降低 `det_size` 到 320
- 切 `model_name = "buffalo_s"`（轻量套件）
- 切 CPU：`device = "cpu"`
- 极限：`detector = "dlib"`（纯 CPU 路径）

### Q3：CPU 部署？

```toml
[face_detection.insightface]
device = "cpu"
```
依赖替换：`onnxruntime-gpu` → `onnxruntime`，`fastdeploy-gpu-python` → `fastdeploy-python`。

### Q4：如何支持十万级人物库？

- 引入 Faiss/Milvus 向量索引代替 `np.dot` 全量计算
- MongoDB 启用 Sharding 按 `number` 分片
- 增量更新：变更时仅更新单个特征向量到索引而非全量重建

### Q5：如何集成到外部系统？

- 直接 HTTP 调用 `/recognize`、`/persons/*`、`/recognize/batch`
- 参考 `app/tests/test_recognize.py` 与 `stress_test_n.py`

### Q6：首次请求仍然很慢？

- 检查启动日志是否出现 `✅ AI 模型预加载完成`
- 看到 `⚠️ AI 模型预加载失败` 时，检查 `ai_models/models/buffalo_l/` 是否完整
- 必要时关掉 `refresh_on_startup` 加速启动（首次识别会延迟加载）

### Q7：InsightFace 还是 dlib？

- **InsightFace（默认）**：GPU 加速，精度更高，适用于生产
- **dlib**：CPU 友好、模型小、稳定可靠，适用于无 GPU 环境
- **混合**：保持 `detector = "insightface"` + 完整模型文件，系统会在 GPU/InsightFace 失败时自动降级到 dlib

### Q8：HTTP 状态码总是 200，怎么判断业务是否成功？

读响应体 `statusCode` 字段：`200` = 真正成功；其他值见上方"状态码总览"。这样可以避免与 HTTP 4xx/5xx 在中间件、网关层冲突。

---

## 测试脚本

| 脚本 | 路径 | 用途 |
|------|------|------|
| 单点识别冒烟 | `app/tests/test_recognize.py` | 验证 `/recognize` 接口可用性 |
| 并发压测 | `app/tests/stress_test_n.py` | N:N 多人脸并发性能（QPS / 响应时间 / 内存） |
| 测试图片 | `app/tests/常泽宇.png`、`app/tests/测试人员.jpg` | 用于本地手动验证 |

---

## 详细文档索引

- 接口详细：`app/docs/RECOGNIZE_API.md` / `PERSONS_API.md` / `OPS_API.md`
- 架构演进：`app/docs/PROJECT_STRUCTURE_OPTIMIZATION.md`
- 优化纪要：`app/docs/OPTIMIZATION_SUMMARY.md`
- 删除接口设计：`app/docs/PERSONS_DELETE_API_OPTIMIZATION.md`

---

## 许可证与联系方式

- Git 仓库：<https://github.com/ZhangShen55/FaceRec>
- 邮箱：seonzheung@gmail.com
- 本项目为内部研发版本，未公开发布前请勿外传。

---

**文档版本**：v3.0（InsightFace 主路径 + Redis Pickle 缓存 + 多帧融合 + 运维监控）
**更新时间**：2026-04
**项目路径**：`/root/workspace/FaceRecAPI_DEV/app`
**主要变更（相对 v2.0）**：
- ✅ 同步实际 `config.toml`：补 `[gpu]`/`[stats]`/`[redis.cache]`/`model_path`/`det_thresh`，修正 section 名（`[threading]`、`[redis.cache.embeddings_ttl]`、`logger.log_path`）
- ✅ 补全 API：`/recognize/batch`、`/persons/batch`、`/persons/search`(POST)、`/persons/delete`、`/ops/health|metrics|stats/*`
- ✅ 重写响应规范：HTTP 永远 200 + `statusCode` 枚举（含 201/202/207/251/252/40x/50x）
- ✅ 新增章节：ROI 多边形掩膜、模型启动预加载、GPU 重映射、API 统计中间件、TTL 集合
- ✅ 数据库设计补 `bbox`、`api_call_logs`、`api_stats_hourly`
- ✅ 修正测试脚本路径、Python 3.10、CUDA 11.8 等环境信息
- ✅ 目录结构按代码实际现状重写
