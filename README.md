# FaceRecAPI 人脸对比服务

基于 FastAPI + Dlib + ArcFace(ONNX/FastDeploy) 的人脸对比系统，提供人员库管理、单图识别、多帧识别、运维统计与简单 Web 管理界面。

## Git 仓库

- https://github.com/ZhangShen55/FaceRec

## 🌟 项目特点

### 核心功能
- **N:N 多人脸识别** ✨：单张图片可同时检测和识别多张人脸，返回所有人脸的 bbox 和匹配结果
- **模型启动预加载** ⚡：Unicorn 启动时自动加载 InsightFace + ArcFace 到 GPU，首请求性能提升 **70-80%**
- **ROI 区域掩膜** 🎯：支持多边形 ROI 掩膜，仅对指定区域内的人脸进行识别

### 技术特性
- Dlib 68 点关键点检测 + 5 点对齐，提升特征稳定性
- ArcFace 512 维特征向量，余弦相似度匹配
- 支持 targets 双阶段匹配（候选阈值 0.2 + 严格阈值 0.4），提高命中率
- 多帧识别聚合，提高视频流/抓拍稳定性
- API 统计中间件 + MongoDB TTL 自动清理

### 系统特性
- 统一的 API 响应格式（HTTP 200 + statusCode 细粒度区分）
- 完整的错误处理和容错机制
- 详细的日志记录和性能统计
- Web 管理界面（HTTP Basic 鉴权）

## 识别流程（/recognize）

### N:N 多人脸识别流程

1. **图片预处理**：Base64 解析 → 像素校验 → ROI 区域掩膜（可选）
2. **人脸检测**：InsightFace 检测所有人脸（返回多个 bbox）
3. **人脸对齐与特征提取**：Dlib 68 点对齐 + ArcFace 提取特征向量
4. **双阶段匹配**：
   - **阶段 1**：对 targets 列表匹配（阈值 0.2，提高召回率）
   - **阶段 2**：对全库匹配（阈值 0.4，保证准确率）
5. **结果聚合**：所有人脸的匹配结果合并去重，按相似度排序
6. **返回结果**：包含所有检测到的人脸的 bbox + 每个人脸的匹配结果

## 目录结构

```
app/
├── main.py                # FastAPI 入口与生命周期管理
├── core/                  # 配置、数据库、AI 引擎、日志
├── router/                # API 路由（faces/persons/ops/web）
├── services/              # 业务服务层（person/ops/face）
├── middleware/            # 统计中间件
├── models/                # 请求/响应模型
├── utils/                 # 图片解析与校验
├── ai_models/             # Dlib/ArcFace 模型文件
├── static/                # Web UI 静态资源
├── media/                 # 人脸裁剪图保存目录
├── logs/                  # 日志输出
└── docs/                  # 详细 API 文档
```

## 快速开始

### 1) 环境要求

- Python 3.10+
- MongoDB 4.4+
- Dlib 编译依赖（cmake, libboost, libopencv）
- GPU 可选（InsightFace 使用，需要 fastdeploy-gpu-python 或仅 CPU 使用 fastdeploy-python）

### 2) 安装依赖

```bash
pip install -r app/requirements.txt
```

### 3) 准备模型文件

将模型文件放入 `app/ai_models/`：

- `shape_predictor_68_face_landmarks.dat`
- `ms1mv3_arcface_r100.onnx`

参考：`app/ai_models/README.md`

### 4) 配置

编辑 `app/config.toml`（按实际环境修改）：

```toml
[db]
username = "root"
password = "root"
host = "10.80.5.25"
port = "27017"
database = "facerecapi"
auth_source = "admin"
limit = 5000

[face]
threshold = 0.4
candidate_threshold = 0.2
rec_min_face_hw = 50

[threading]
max_workers = 2

[frontlogin]
username = "admin"
password = "admin"

[image]
max_feature_image_width_px = 9999
max_feature_image_height_px = 9999
min_feature_image_width_px = 80
min_feature_image_height_px = 80
max_feature_image_size_m = 10
max_face_hw = 999
min_face_hw = 50

[stats]
retention_days = 7
hourly_retention_days = 30
```

### 5) 启动服务

在项目根目录执行（注：首次启动会预加载模型，耗时 15-30 秒）：

```bash
cd /root/workspace/FaceRecAPI_DEV/app
PYTHONPATH=/root/workspace/FaceRecAPI_DEV OMP_NUM_THREADS=1 \
uvicorn app.main:app --host 0.0.0.0 --port 8003 --workers 1
```

**预期启动日志**（模型预加载过程）：

```
✅ MongoDB ping ok
✅ Redis 连接成功
✅ 启动时已加载 XXX 个人员特征到 Redis
正在初始化 Dlib 进程池，工作线程数: 2...
✅ Dlib 进程池初始化完成
🔄 预加载 AI 模型到 GPU...
🔄 预加载 InsightFace 检测模型...
✅ InsightFace 已预加载
🔄 预加载 Embedding 模型 (ArcFace)...
✅ Embedding 模型已预加载到 GPU
✅ AI 模型预加载完成
Uvicorn running on http://0.0.0.0:8003
```

**Swagger UI**: `http://localhost:8003/docs`

### 📊 性能指标（模型预加载后）

| 指标 | 数值 | 说明 |
|------|------|------|
| 启动时间 | 15-30 秒 | 包含 MongoDB、Redis、特征缓存加载和模型初始化 |
| 首次识别请求 | 2-3 秒 | 模型已预加载，相比无预加载的 10-15 秒提升 **70-80%** |
| 后续请求 | 200-500ms | 常规识别时间 |
| GPU 显存占用 | 6-9 GB | InsightFace (4-6GB) + ArcFace (2-3GB) |
| N:N 多人脸 | 支持 | 单张图可检测 5-10+ 张人脸 |

**GPU 兼容性**：
- RTX 3090 (24GB)：✅ 推荐
- RTX 4090 (24GB)：✅ 推荐
- A100 (40GB)：✅ 最佳
- T4 (16GB)：⚠️ 可用但显存紧张

## API 速览

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/persons` | 新增或更新人物（按 number 去重） |
| POST | `/persons/batch` | 批量新增/更新人物 |
| GET | `/persons` | 人物列表（分页） |
| POST | `/persons/search` | 搜索人物（name 模糊/number 精确） |
| DELETE | `/persons/delete` | 通用删除（name/number/id） |
| POST | `/recognize` | 单图识别 |
| POST | `/recognize/batch` | 多帧识别并聚合 |
| GET | `/ops/health` | 健康检查 |
| GET | `/ops/metrics` | 系统指标 |
| GET | `/ops/stats/api-calls` | API 调用明细 |
| GET | `/ops/stats/hourly` | 按小时统计 |
| GET | `/ops/stats/summary` | 汇总统计 |
| GET | `/` | Web 管理页（HTTP Basic） |

详细接口文档：

- `app/docs/PERSONS_API.md`
- `app/docs/RECOGNIZE_API_ERRORS.md`
- `app/docs/OPS_API.md`

## 请求示例

### 1) 添加人物

```json
POST /persons
{
  "name": "张三",
  "number": "T001",
  "photo": "data:image/jpeg;base64,/9j/4AAQSkZJRg..."
}
```

### 2) 单图识别

```json
POST /recognize
{
  "photo": "data:image/jpeg;base64,/9j/4AAQSkZJRg...",
  "targets": ["T001", "T002"],
  "threshold": 0.4,
  "points":[
        {"x":0, "y":0},{"x":1910, "y":0},
        {"x":1910, "y":540},{"x":0, "y":540}]
}
```

**响应示例**（多人脸检测）：

```json
{
  "statusCode": 200,
  "message": "识别成功",
  "data": {
    "hasFace": true,
    "bboxs": [
      {"x": 100, "y": 120, "w": 150, "h": 150},
      {"x": 250, "y": 100, "w": 160, "h": 160}
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
    "message": "检测到2个人脸，匹配成功"
  }
}
```

返回的 `match` 为相似度降序列表，包含 `is_target` 标记。
`points` 参数用于 ROI 多边形掩膜（可选），仅识别多边形内的人脸。

## 数据存储

### persons 集合

- `_id`: ObjectId
- `name`: 人物姓名
- `number`: 唯一编号
- `photo_path`: 裁剪人脸图片路径
- `bbox`: 人脸框字符串 `x,y,w,h`
- `embedding`: 512 维特征向量（Binary, float32 bytes）
- `tip`: 图像质量提示

### 统计集合

- `api_call_logs`: 详细请求日志（TTL 清理）
- `api_stats_hourly`: 按小时聚合统计（TTL 清理）

## 📌 说明与注意事项

### 数据格式
- `photo` 字段必须是 `data:image/...;base64,` 前缀的 Base64 字符串
- `targets` 为人员编号列表（`number`），匹配分两个阶段：候选阶段阈值 0.2，严格阶段阈值 0.4
- `points` 参数为 ROI 多边形顶点坐标列表，用于掩膜指定区域（可选）

### 核心功能说明

#### N:N 多人脸识别
- **特点**：支持单张图片中的多张人脸同时检测和识别
- **响应**：`bboxs` 字段包含所有检测到的人脸（数组格式），每个人脸都进行匹配
- **场景**：人群识别、多人通行等
- **限制**：一般支持 5-10+ 张人脸，具体数量取决于图片分辨率和人脸清晰度

#### 模型启动预加载
- **工作原理**：系统启动时自动加载 InsightFace 和 ArcFace 到 GPU，无需等待首次请求
- **性能提升**：首请求从 10-15 秒降低到 2-3 秒（提升 70-80%）
- **容错机制**：如果预加载失败，系统仍继续启动，首次请求时进行延迟加载
- **GPU 显存**：占用 6-9 GB（InsightFace 4-6GB + ArcFace 2-3GB）

#### ROI 区域掩膜
- **功能**：使用多边形顶点定义关注区域，仅识别区域内的人脸
- **坐标系**：左上角 (0,0)，坐标范围 [0, 图片宽度-1] × [0, 图片高度-1]
- **顶点要求**：≥3 个点，按顺时针顺序（Top-Left → Top-Right → Bottom-Right → Bottom-Left）
- **自动功能**：超出边界的坐标自动裁剪到图片范围内
- **返回值**：`bboxs` 坐标相对于**原始图片**，不是 ROI 区域

### 配置说明
- Dlib 检测使用进程池，`threading.max_workers` 建议 1~2（防止内存溢出）
- InsightFace 检测阈值由 `det_thresh` 控制，可在 config.toml 中调整
- 日志由 `LOG_LEVEL` 与 `LOG_DIR` 环境变量控制，默认写入 `/app/logs/facerecapi.log`
- Web 管理页 `/` 使用 HTTP Basic 鉴权，账号密码来自 `frontlogin` 配置

### 推荐部署

**开发/测试环境**：
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8003 --workers 1 --reload
```

**生产环境**：
```bash
# 后台启动（推荐使用 systemd 或 supervisor）
cd /root/workspace/FaceRecAPI_DEV/app
nohup env PYTHONPATH=/root/workspace/FaceRecAPI_DEV OMP_NUM_THREADS=1 \
  uvicorn app.main:app --host 0.0.0.0 --port 8003 --workers 1 \
  > logs/facerec_server.log 2>&1 &
```

### 监控和日志
- 查看启动日志：`tail -f /root/workspace/FaceRecAPI_DEV/app/logs/facerecapi.log`
- 监控 GPU：`nvidia-smi` 或 `watch -n 2 nvidia-smi`
- 健康检查：`curl http://localhost:8003/ops/health`
- 系统指标：`curl http://localhost:8003/ops/metrics`

### 故障排除

**首次请求仍然很慢**
- 检查启动日志是否看到 `✅ AI 模型预加载完成`
- 如果有 `⚠️ AI 模型预加载失败`，说明模型未预加载，首次请求会进行延迟加载

**GPU 显存未增加**
- 等待启动完成后 30 秒再检查 nvidia-smi
- 检查 config.toml 中的 gpu_id 是否正确
- 运行 `nvidia-smi -q` 查看 GPU 详细状态

**识别准确率不理想**
- 检查图片质量：光线充足、人脸清晰、无遮挡
- 调整阈值：提高阈值减少误匹配，降低阈值提高召回率
- 重新录入人员库：确保初始特征高质量

## 联系方式

- 邮箱: seonzheung@gmail.com
