# Recognize API 文档

人脸识别接口的完整说明文档。

**最新更新**：v6.0 (2026-01-19) - 新增 N:N 多人脸识别、模型启动预加载、ROI 掩膜功能

---

## 📋 目录

- [重要说明](#重要说明)
- [关键功能更新](#关键功能更新)
- [接口概览](#接口概览)
- [统一响应格式](#统一响应格式)
- [状态码说明](#状态码说明)
- [单张图片识别](#单张图片识别)
- [性能指标](#性能指标)
- [最佳实践](#最佳实践)

---

## 重要说明

### 🔥 v6.0 重大功能更新（2026-01-19）

本版本新增 3 个核心功能，显著提升性能和功能完整性：

#### 1️⃣ **模型启动时预加载** ⚡

系统启动时自动加载 InsightFace 和 Embedding 模型到 GPU

**性能对比**：
| 指标 | 无预加载 | 有预加载 | 改进 |
|------|---------|--------|------|
| 首次请求 | 10-15s | 2-3s | **70-80%** |
| 启动时间 | ~5s | 15-30s | +10-25s |
| GPU 显存 | 无占用 | 6-9GB | 启动后立即可用 |

**启动日志示例**：
```
✅ Dlib 进程池初始化完成
🔄 预加载 AI 模型到 GPU...
✅ InsightFace 已预加载
✅ Embedding 模型已预加载到 GPU
✅ AI 模型预加载完成
Uvicorn running on http://0.0.0.0:8003
```

#### 2️⃣ **N:N 多人脸识别** ✨

单张图片可同时检测和识别多张人脸

- 原来：`bbox` 字段（单人脸对象）
- 现在：`bboxs` 字段（多人脸数组）

#### 3️⃣ **ROI 多边形掩膜** 🎯

使用多边形定义关注区域，仅识别区域内的人脸

---

## 关键功能更新

### 模型预加载工作原理

启动时自动加载模型，首次请求响应时间从 10-15s 降低到 2-3s（**提升 70-80%**）。

预加载失败时系统仍继续启动，首次请求时进行延迟加载。

### N:N 多人脸识别

支持单张图片中的多张人脸同时检测和识别。

一般支持 5-10+ 张人脸，具体数量取决于图片分辨率和人脸清晰度。

### ROI 区域掩膜

使用多边形顶点定义关注区域：
- 点数 ≥ 3 个，按顺时针顺序
- 坐标自动裁剪到图片范围内
- `bboxs` 坐标相对于原始图片

---

## 接口概览

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/recognize` | 单张图片人脸识别（支持单人脸和多人脸） |
| POST | `/recognize/batch` | 批量识别（多帧独立识别并聚合） |

---

## 统一响应格式

所有接口都返回以下格式（HTTP 状态码永远是 200）：

```json
{
  "statusCode": 200,
  "message": "操作成功",
  "data": {
    // 具体数据，根据接口不同而不同
  }
}
```

**字段说明**：

| 字段 | 类型 | 说明 |
|------|------|------|
| statusCode | int | 业务状态码 |
| message | string | 操作结果描述信息 |
| data | object/null | 成功时包含具体数据，失败时可能为 null |

---

## 状态码说明

| statusCode | 含义 | 场景 |
|-------------|------|------|
| **200** | 识别成功 | match 不为空 |
| **201** | 未检测到人脸 | 图片有效但无人脸 |
| **202** | 人脸尺寸过小 | 检测到但尺寸不足 |
| **251** | 数据库为空 | 检测到人脸但数据库无数据 |
| **252** | 未匹配到对象 | 相似度低于阈值 |
| **401** | base64 解码失败 | base64 格式错误 |
| **402** | 图片格式错误 | cv2 无法解析 |
| **403** | 无效图片数据 | image_data 为 None/空 |
| **501** | 检测服务错误 | AI 引擎异常 |
| **502** | 特征提取失败 | 特征提取服务异常 |

---

## 单张图片识别

### `POST /recognize`

**请求体**：

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

**字段说明**：

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| photo | string | 是 | Base64编码的图片数据 |
| targets | array | 否 | 优先匹配的人员编号列表 |
| threshold | float | 否 | 识别阈值 |
| points | array | 否 | ROI 多边形顶点列表 |

---

### 响应示例

**成功识别（多人脸）** → statusCode=200：

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

**未检测到人脸** → statusCode=201：

```json
{
  "statusCode": 201,
  "message": "图像中未检测到人脸，请重新捕捉人脸",
  "data": null
}
```

**人脸尺寸过小** → statusCode=202：

```json
{
  "statusCode": 202,
  "message": "人脸像素过小(60x60px)，无法识别",
  "data": {
    "hasFace": true,
    "bboxs": [{"x": 100, "y": 150, "w": 60, "h": 60}],
    "threshold": 0.4,
    "match": null
  }
}
```

**数据库为空** → statusCode=251：

```json
{
  "statusCode": 251,
  "message": "数据库为空，请先录入人员信息",
  "data": {
    "hasFace": true,
    "bboxs": [{"x": 100, "y": 150, "w": 200, "h": 200}],
    "match": null
  }
}
```

**未匹配到对象** → statusCode=252：

```json
{
  "statusCode": 252,
  "message": "未找到匹配的人物（相似度低于阈值）",
  "data": {
    "hasFace": true,
    "bboxs": [{"x": 100, "y": 150, "w": 200, "h": 200}],
    "threshold": 0.4,
    "match": null
  }
}
```

---

## 性能指标

### 启动性能（带模型预加载）

| 指标 | 数值 | 说明 |
|------|------|------|
| 系统启动时间 | 15-30 秒 | 包含所有初始化 |
| └─ 模型预加载 | 10-20 秒 | InsightFace + ArcFace |
| GPU 显存占用 | 6-9 GB | 启动后立即占用 |

### 请求性能（模型预加载后）

| 指标 | 数值 | 说明 |
|------|------|------|
| 首次请求 | 2-3 秒 | 相比无预加载 10-15s 提升 70-80% |
| 后续请求 | 200-500 ms | 常规识别 |
| N:N 多人脸 | +30-50 ms/人脸 | 相比单人脸的增加 |

---

## 最佳实践

### 1. 单张图片识别

```javascript
async function recognizeFace(photo, targets = [], threshold = null) {
  const response = await fetch('/recognize', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ photo, targets, threshold })
  });

  const result = await response.json();

  switch (result.statusCode) {
    case 200:
      return {
        success: true,
        matches: result.data.match,
        faceCount: result.data.bboxs.length  // ✨ 多人脸支持
      };
    case 201:
      return { success: false, reason: 'no_face' };
    case 202:
      return { success: false, reason: 'face_too_small' };
    case 251:
      return { success: false, reason: 'db_empty' };
    case 252:
      return { success: false, reason: 'no_match' };
    default:
      return { success: false, reason: 'error', message: result.message };
  }
}
```

### 2. ROI 多边形掩膜

```javascript
// 定义监控区域（例如：门禁区域）
const monitoringArea = [
  { X: 0, Y: 0 },          // 左上角
  { X: 1920, Y: 0 },       // 右上角
  { X: 1920, Y: 540 },     // 右下角
  { X: 0, Y: 540 }         // 左下角
];

async function recognizeInRegion(photo) {
  const response = await fetch('/recognize', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      photo: photo,
      points: monitoringArea,
      threshold: 0.4
    })
  });

  return await response.json();
}
```

---

## 📝 注意事项

1. **N:N 识别**：`bboxs` 为数组格式，支持多人脸，每个人脸都进行完整的特征提取和匹配
2. **ROI 掩膜**：`points` 参数为 ROI 多边形顶点，坐标自动裁剪到图片范围，返回的 bbox 相对于原始图片
3. **模型预加载**：启动后自动加载完成，首次请求无需等待，提升 70-80% 性能
4. **targets 匹配阈值**：使用 `threshold / 2` 作为候选阈值
5. **match 列表排序**：按相似度降序排列，`match[0]` 即最相似结果

---

## 更新日志

| 版本 | 日期 | 变更内容 |
|------|------|---------|
| v6.0 | 2026-01-19 | 新增模型启动预加载（70-80% 性能提升）、N:N 多人脸识别、ROI 掩膜 |
| v5.0 | 2026-01-13 | 统一响应格式，HTTP 状态码永远 200，通过 statusCode 区分成功/失败 |
| v2.1 | 2026-01-09 | 更新响应结构（match 列表 + is_target） |
| v2.0 | 2026-01-06 | 统一返回 HTTP 200 + 结构化响应 |
| v1.0 | - | 初始版本 |

---

## 相关文档

- [README.md](../README.md) - 项目概述和快速开始
- [PERSONS_API.md](./PERSONS_API.md) - 人物管理接口文档
- [MODEL_PRELOADING_GUIDE.md](../../MODEL_PRELOADING_GUIDE.md) - 模型预加载详细指南
- [ROI_USAGE_GUIDE.md](../../ROI_USAGE_GUIDE.md) - ROI 功能使用指南

---

## 📞 技术支持

邮箱：<seonzheung@gmail.com>
