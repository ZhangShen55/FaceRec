# `/recognize` 接口错误码说明文档

## 📌 接口概述

**接口路径**:
- `POST /recognize`：单张图片识别
- `POST /recognize/batch`：多帧独立识别并聚合结果

**功能**: 人脸识别接口，识别上传图片中的人脸是否与数据库中已知人物匹配。

---

## ✅ `/recognize` 请求参数

```json
{
  "photo": "data:image/jpeg;base64,/9j/4AAQSkZJRg...",
  "targets": ["T001", "T002"],
  "threshold": 0.4
}
```

- `photo`: Base64 编码图片
- `targets`: **人员编号列表**（可选）
- `threshold`: 识别阈值（可选，默认使用配置文件 `face.threshold`）

---

## ✅ `/recognize` 成功响应 (HTTP 200)

```json
{
  "has_face": true,
  "bbox": {
    "x": 100,
    "y": 150,
    "w": 200,
    "h": 200
  },
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
  "message": "匹配成功，≥阈值40.00%有2位，targets命中1位，最相似的是张三_T001"
}
```

**字段说明**:
- `match`: 可能是 `null` 或列表，按相似度降序排列
- `similarity`: 百分比字符串
- `is_target`: 是否来自 `targets` 优先匹配结果

### 业务场景详解

| 场景 | `has_face` | `match` | `message` 示例 |
|------|-----------|---------|---------------|
| ✅ 匹配成功 | `true` | 列表 | "匹配成功，≥阈值40.00%有2位，最相似的是张三_T001" |
| ⚠️ 匹配失败（相似度低） | `true` | `null` | "匹配失败，未能够匹配到目标人物" |
| ⚠️ 未检测到人脸 | `false` | `null` | "图像中未检测到人脸，请重新捕捉人脸" |
| ⚠️ 人脸过小 | `true` | `null` | "人脸像素过小(60x60px)，无法识别" |
| ⚠️ 数据库无数据 | `true` | `null` | "匹配失败，未能够匹配到目标人物" |

---

## ❌ `/recognize` 错误响应 (HTTP 4xx/5xx)

仅在**真正错误**（请求参数错误、服务故障）时返回异常状态码。

### HTTP 400 - Bad Request

**触发条件**: 图片数据无效或为空

```json
{
  "detail": "[recognize] 未接收到有效图片数据或图像数据存在异常"
}
```

### HTTP 500 - Internal Server Error

**触发条件**: AI 引擎或特征提取异常

```json
{
  "detail": "[recognize] 人脸检测服务内部错误"
}
```

---

## ✅ `/recognize/batch` 请求参数

```json
{
  "photos": [
    "data:image/jpeg;base64,/9j/4AAQSkZJRg...",
    "data:image/jpeg;base64,/9j/4AAQSkZJRg..."
  ],
  "targets": ["T001", "T002"],
  "threshold": 0.4
}
```

- `photos`: Base64 图片列表（多帧）
- `targets`: **人员编号列表**（可选）
- `threshold`: 识别阈值（可选）

---

## ✅ `/recognize/batch` 成功响应 (HTTP 200)

```json
{
  "total_frames": 2,
  "valid_frames": 1,
  "threshold": 0.4,
  "frames": [
    {"index": 0, "has_face": true, "bbox": {"x": 100, "y": 120, "w": 200, "h": 200}, "error": null},
    {"index": 1, "has_face": false, "bbox": null, "error": "未检测到人脸"}
  ],
  "match": [
    {
      "id": "507f1f77bcf86cd799439011",
      "name": "张三",
      "number": "T001",
      "similarity": "86.20%",
      "is_target": false
    }
  ],
  "confidence": 0.5,
  "message": "识别成功，使用1帧有效图片，找到1位候选人，最相似的是张三_T001（出现1次）"
}
```

**字段说明**:
- `frames`: 每帧处理结果（仅包含是否检测到人脸/错误信息）
- `match`: 聚合后的 top3 结果（按最高相似度排序）
- `confidence`: `valid_frames / total_frames`

---

## ❌ `/recognize/batch` 错误响应 (HTTP 4xx/5xx)

### HTTP 400 - Bad Request

**触发条件**: `photos` 为空

```json
{
  "detail": "photos 列表不能为空"
}
```

### HTTP 500 - Internal Server Error

**触发条件**: 未捕获的服务异常

---

## 🛠️ 客户端错误处理建议

```javascript
async function recognizeFace(photo, targets = [], threshold = null) {
  const response = await fetch('/recognize', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ photo, targets, threshold })
  });

  if (!response.ok) {
    const error = await response.json();
    return { error: error.detail || '服务异常' };
  }

  const result = await response.json();

  if (!result.has_face) {
    return { error: '未检测到人脸，请重新拍照' };
  }

  if (result.match && result.match.length > 0) {
    const best = result.match[0];
    return {
      success: true,
      person: best,
      message: result.message
    };
  }

  return { success: false, message: result.message || '未匹配到已知人物' };
}
```

---

## 📝 注意事项

1. `targets` 目前只接收 **人员编号列表**，不接收 `{name, number}` 结构。
2. `targets` 命中时使用 **`threshold / 2`** 作为候选阈值；响应中的 `threshold` 仍是全局阈值。
3. `match` 列表按相似度降序排列，`match[0]` 即最相似结果。
4. `similarity` 为百分比字符串，便于直接展示。

---

## 🔄 更新日志

| 版本 | 日期 | 变更内容 |
|------|------|---------|
| v2.1 | 2026-01-09 | 更新响应结构（match 列表 + is_target），补充 batch 接口说明 |
| v2.0 | 2026-01-06 | 统一返回 HTTP 200 + 结构化响应 |
| v1.0 | - | 初始版本 |

---

## 📞 技术支持

如遇问题，请提供以下信息：
- 请求的完整 JSON
- 返回的 HTTP 状态码和响应体
- 后端日志中的 `[recognize]` 相关日志
