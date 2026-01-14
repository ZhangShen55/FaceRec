# Recognize API 文档

人脸识别接口的完整说明文档。

---

## 📋 目录

- [重要说明](#重要说明)
- [接口概览](#接口概览)
- [统一响应格式](#统一响应格式)
- [状态码说明](#状态码说明)
- [单张图片识别](#单张图片识别)
- [批量识别](#批量识别)
- [最佳实践](#最佳实践)

---

## 重要说明

**🔥 v5.0 重大变更**

从 v5.0 开始，所有接口遵循以下规则：

1. **HTTP 状态码永远是 200** - 不再抛出 HTTP 异常
2. **通过 `statusCode` 字段判断结果** - 成功/失败都在响应体中
3. **统一响应格式** - 所有接口返回相同结构：`{statusCode, message, data}`
4. **细粒度的状态码** - 区分不同的错误场景（图片解析错误、人脸检测失败、未匹配等）

**前端/客户端适配要点：**
```javascript
// ❌ 旧方式（v4.0 及之前）
try {
  const response = await fetch('/recognize', {...});
  if (!response.ok) throw new Error('HTTP error');
  const data = await response.json();
} catch (error) {
  // 处理 HTTP 异常
}

// ✅ 新方式（v5.0）
const response = await fetch('/recognize', {...});
const result = await response.json();  // HTTP 永远是 200

if (result.statusCode === 200) {
  // 成功 - 匹配到人物
  console.log('匹配成功:', result.data.match);
} else if (result.statusCode === 201) {
  // 未检测到人脸
  console.warn('未检测到人脸');
} else if (result.statusCode === 202) {
  // 人脸过小
  console.warn('人脸过小');
} else if (result.statusCode === 251) {
  // 数据库为空
  console.warn('数据库为空');
} else if (result.statusCode === 252) {
  // 未匹配到对象
  console.warn('未匹配到对象');
} else {
  // 其他错误
  console.error(result.message);
}
```

---

## 接口概览

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/recognize` | 单张图片人脸识别 |
| POST | `/recognize/batch` | 批量识别（多帧独立识别并聚合） |

---

## 统一响应格式

**所有接口**都返回以下格式（HTTP 状态码永远是 200）：

```json
{
  "statusCode": 200,
  "message": "操作成功",
  "data": {
    // 具体数据，根据接口不同而不同
  }
}
```

**字段说明：**

| 字段 | 类型 | 说明 |
|------|------|------|
| statusCode | int | 业务状态码（200=成功且匹配，201/202/251/252=其他场景，4xx/5xx=错误） |
| message | string | 操作结果描述信息 |
| data | object/null | 成功时包含具体数据，失败时可能为 null 或包含错误详情 |

---

## 状态码说明

### 成功/部分成功（2xx）

| statusCode | 含义 | 适用场景 | data 是否为空 |
|-------------|------|---------|--------------|
| **200** | 识别成功且匹配到人物 | match 不为空 | ✅ 有 data |
| **201** | 未检测到人脸 | 图片有效但无人脸 | ❌ data 为 null |
| **202** | 人脸尺寸过小 | 有 bbox 但 match 为空 | ✅ 有 data |

### 数据库相关（25x）

| statusCode | 含义 | 适用场景 | data 是否为空 |
|-------------|------|---------|--------------|
| **251** | 数据库为空 | 有 bbox 但 match 为空 | ✅ 有 data |
| **252** | 未匹配到对象 | 相似度低于阈值 | ✅ 有 data |

### 客户端错误 - 图片数据相关（40x）

| statusCode | 含义 | 适用场景 | data 是否为空 |
|-------------|------|---------|--------------|
| **400** | 通用请求参数错误 | photos列表为空等 | ❌ data 为 null |
| **401** | base64 解码失败 | base64 格式错误 | ❌ data 为 null |
| **402** | 图片格式错误 | cv2 无法解析 | ❌ data 为 null |
| **403** | 未接收到有效图片数据 | image_data 为 None/空 | ❌ data 为 null |

### 服务器错误（5xx）

| statusCode | 含义 | 适用场景 | data 是否为空 |
|-------------|------|---------|--------------|
| **501** | 人脸检测服务内部错误 | AI 引擎异常 | ❌ data 为 null |
| **502** | 特征提取失败 | 特征提取服务异常 | ❌ data 为 null |

---

## 单张图片识别

### `POST /recognize`

识别上传图片中的人脸是否与数据库中已知人物匹配。

**请求体**：

```json
{
  "photo": "data:image/jpeg;base64,/9j/4AAQSkZJRg...",
  "targets": ["T001", "T002"],
  "threshold": 0.4
}
```

**字段说明**：

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| photo | string | 是 | Base64编码的图片数据 |
| targets | array | 否 | 优先匹配的人员编号列表（使用 threshold/2 作为候选阈值） |
| threshold | float | 否 | 识别阈值（默认使用配置文件中的值） |

---

### 响应场景详解

#### **场景 1：识别成功且匹配到人物** → statusCode=200

**条件**：`data.match` 不为空

```json
{
  "statusCode": 200,
  "message": "识别成功",
  "data": {
    "hasFace": true,
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
}
```

**data 字段说明**：

| 字段 | 类型 | 说明 |
|------|------|------|
| hasFace | boolean | 是否检测到人脸 |
| bbox | object | 人脸框坐标 {x, y, w, h} |
| threshold | float | 使用的识别阈值 |
| match | array | 匹配结果列表（最多3个，按相似度降序） |
| message | string | 详细的识别结果描述 |

**match 数组元素说明**：

| 字段 | 类型 | 说明 |
|------|------|------|
| id | string | 人物数据库ID |
| name | string | 人物姓名 |
| number | string | 人物编号 |
| similarity | string | 相似度百分比（如 "87.45%"） |
| is_target | boolean | 是否来自 targets 优先匹配 |

---

#### **场景 2：未检测到人脸** → statusCode=201

**条件**：图片有效但未检测到人脸

```json
{
  "statusCode": 201,
  "message": "图像中未检测到人脸，请重新捕捉人脸",
  "data": null
}
```

---

#### **场景 3：人脸尺寸过小** → statusCode=202

**条件**：检测到人脸但尺寸不足

```json
{
  "statusCode": 202,
  "message": "人脸像素过小(60x60px)，无法识别",
  "data": {
    "hasFace": true,
    "bbox": {
      "x": 100,
      "y": 150,
      "w": 60,
      "h": 60
    },
    "threshold": 0.4,
    "match": null,
    "message": "人脸像素过小(60x60px)，无法识别"
  }
}
```

---

#### **场景 4：数据库为空** → statusCode=251

**条件**：检测到人脸但数据库中没有注册人员

```json
{
  "statusCode": 251,
  "message": "数据库为空，请先录入人员信息",
  "data": {
    "hasFace": true,
    "bbox": {
      "x": 100,
      "y": 150,
      "w": 200,
      "h": 200
    },
    "threshold": 0.4,
    "match": null,
    "message": "数据库为空，请先录入人员信息"
  }
}
```

---

#### **场景 5：未匹配到对象（相似度低于阈值）** → statusCode=252

**条件**：检测到人脸且数据库有数据，但相似度均低于阈值

```json
{
  "statusCode": 252,
  "message": "未找到匹配的人物（相似度低于阈值）",
  "data": {
    "hasFace": true,
    "bbox": {
      "x": 100,
      "y": 150,
      "w": 200,
      "h": 200
    },
    "threshold": 0.4,
    "match": null,
    "message": "未找到匹配的人物（相似度低于阈值）"
  }
}
```

---

#### **场景 6：base64 解码失败** → statusCode=401

```json
{
  "statusCode": 401,
  "message": "base64 解码失败: Incorrect padding",
  "data": null
}
```

---

#### **场景 7：图片格式错误** → statusCode=402

```json
{
  "statusCode": 402,
  "message": "无法解析图片格式",
  "data": null
}
```

---

#### **场景 8：未接收到有效图片数据** → statusCode=403

```json
{
  "statusCode": 403,
  "message": "未接收到有效图片数据或图像数据存在异常",
  "data": null
}
```

---

#### **场景 9：人脸检测服务内部错误** → statusCode=501

```json
{
  "statusCode": 501,
  "message": "人脸检测服务内部错误: <具体错误信息>",
  "data": null
}
```

---

#### **场景 10：特征提取失败** → statusCode=502

```json
{
  "statusCode": 502,
  "message": "人脸特征提取失败: <具体错误信息>",
  "data": null
}
```

---

## 批量识别

### `POST /recognize/batch`

批量识别接口（多帧独立识别，取最优结果）。

**功能说明**：
- 每张图片独立识别
- 汇总所有结果取最高相似度
- 适用场景：视频流抓拍、同一人的多角度照片等

**请求体**：

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

**字段说明**：

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| photos | array | 是 | Base64编码的图片列表（不能为空） |
| targets | array | 否 | 优先匹配的人员编号列表 |
| threshold | float | 否 | 识别阈值（默认使用配置文件中的值） |

---

### 响应场景详解

#### **场景 1：识别成功且匹配到人物** → statusCode=200

```json
{
  "statusCode": 200,
  "message": "批量识别成功",
  "data": {
    "total_frames": 2,
    "valid_frames": 2,
    "threshold": 0.4,
    "frames": [
      {
        "index": 0,
        "hasFace": true,
        "bbox": {"x": 100, "y": 120, "w": 200, "h": 200},
        "error": null
      },
      {
        "index": 1,
        "hasFace": true,
        "bbox": {"x": 110, "y": 130, "w": 190, "h": 190},
        "error": null
      }
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
    "message": "识别成功，使用2帧有效图片，找到1位候选人，最相似的是张三_T001（出现2次）"
  }
}
```

**data 字段说明**：

| 字段 | 类型 | 说明 |
|------|------|------|
| total_frames | int | 总帧数 |
| valid_frames | int | 有效帧数（成功提取特征的帧数） |
| threshold | float | 使用的识别阈值 |
| frames | array | 每帧的处理结果 |
| match | array | 聚合后的 top3 结果（按最高相似度排序） |
| message | string | 详细的识别结果描述 |

**frames 数组元素说明**：

| 字段 | 类型 | 说明 |
|------|------|------|
| index | int | 帧索引（从0开始） |
| hasFace | boolean | 是否检测到人脸 |
| bbox | object/null | 人脸框坐标（如果有） |
| error | string/null | 错误信息（如果有） |

---

#### **场景 2：photos 列表为空** → statusCode=400

```json
{
  "statusCode": 400,
  "message": "photos 列表不能为空",
  "data": null
}
```

---

#### **场景 3：数据库为空** → statusCode=251

```json
{
  "statusCode": 251,
  "message": "数据库为空，请先录入人员信息",
  "data": {
    "total_frames": 2,
    "valid_frames": 0,
    "threshold": 0.4,
    "frames": [],
    "match": null,
    "message": "数据库中暂无人脸数据，请先录入"
  }
}
```

---

#### **场景 4：所有帧均未检测到有效人脸** → statusCode=201

```json
{
  "statusCode": 201,
  "message": "所有帧均未检测到有效人脸",
  "data": {
    "total_frames": 2,
    "valid_frames": 0,
    "threshold": 0.4,
    "frames": [
      {
        "index": 0,
        "hasFace": false,
        "bbox": null,
        "error": "未检测到人脸"
      },
      {
        "index": 1,
        "hasFace": false,
        "bbox": null,
        "error": "未检测到人脸"
      }
    ],
    "match": null,
    "message": "所有帧均未检测到有效人脸"
  }
}
```

---

#### **场景 5：未匹配到对象** → statusCode=252

```json
{
  "statusCode": 252,
  "message": "识别失败，使用2帧有效图片，但相似度均低于阈值",
  "data": {
    "total_frames": 2,
    "valid_frames": 2,
    "threshold": 0.4,
    "frames": [
      {
        "index": 0,
        "hasFace": true,
        "bbox": {"x": 100, "y": 120, "w": 200, "h": 200},
        "error": null
      },
      {
        "index": 1,
        "hasFace": true,
        "bbox": {"x": 110, "y": 130, "w": 190, "h": 190},
        "error": null
      }
    ],
    "match": null,
    "message": "识别失败，使用2帧有效图片，但相似度均低于阈值"
  }
}
```

---

## 最佳实践

### 1. 统一的错误处理

```javascript
async function callRecognizeApi(url, options) {
  const response = await fetch(url, options);
  const result = await response.json();  // HTTP 永远是 200

  if (result.statusCode === 200) {
    return { success: true, data: result.data };
  } else {
    return { success: false, statusCode: result.statusCode, message: result.message };
  }
}
```

### 2. 单张图片识别

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
      // 成功匹配
      return {
        success: true,
        person: result.data.match[0],
        allMatches: result.data.match,
        message: result.data.message
      };

    case 201:
      // 未检测到人脸
      return { success: false, reason: 'no_face', message: '未检测到人脸，请重新拍照' };

    case 202:
      // 人脸过小
      return { success: false, reason: 'face_too_small', message: '人脸过小，请靠近镜头' };

    case 251:
      // 数据库为空
      return { success: false, reason: 'db_empty', message: '系统中暂无注册人员' };

    case 252:
      // 未匹配到对象
      return { success: false, reason: 'no_match', message: '未识别到已知人物' };

    case 401:
    case 402:
    case 403:
      // 图片数据错误
      return { success: false, reason: 'invalid_image', message: '图片数据错误，请重新上传' };

    case 501:
    case 502:
      // 服务错误
      return { success: false, reason: 'service_error', message: '服务暂时不可用，请稍后重试' };

    default:
      return { success: false, reason: 'unknown', message: result.message };
  }
}
```

### 3. 批量识别

```javascript
async function batchRecognize(photos, targets = [], threshold = null) {
  const response = await fetch('/recognize/batch', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ photos, targets, threshold })
  });

  const result = await response.json();

  if (result.statusCode === 200) {
    // 识别成功
    return {
      success: true,
      person: result.data.match[0],
      allMatches: result.data.match,
      validFrames: result.data.valid_frames,
      totalFrames: result.data.total_frames,
      message: result.data.message
    };
  } else if (result.statusCode === 201) {
    // 所有帧均无人脸
    return { success: false, reason: 'no_face', message: '所有图片均未检测到人脸' };
  } else if (result.statusCode === 251) {
    // 数据库为空
    return { success: false, reason: 'db_empty', message: '系统中暂无注册人员' };
  } else if (result.statusCode === 252) {
    // 未匹配
    return { success: false, reason: 'no_match', message: '未识别到已知人物' };
  } else {
    return { success: false, reason: 'error', message: result.message };
  }
}
```

### 4. React 示例（使用 hooks）

```javascript
import { useState } from 'react';

function FaceRecognition() {
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  const handleRecognize = async (photoBase64) => {
    setError(null);

    const response = await fetch('/recognize', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ photo: photoBase64 })
    });

    const data = await response.json();

    if (data.statusCode === 200) {
      setResult(data.data);
      alert(`识别成功: ${data.data.match[0].name}`);
    } else if (data.statusCode === 201) {
      setError('未检测到人脸，请重新拍照');
    } else if (data.statusCode === 252) {
      setError('未识别到已知人物');
    } else {
      setError(data.message);
    }
  };

  return (
    <div>
      {error && <div className="error">{error}</div>}
      {result && (
        <div className="success">
          识别到: {result.match[0].name} ({result.match[0].similarity})
        </div>
      )}
      {/* ... 拍照组件 */}
    </div>
  );
}
```

---

## 两个接口的状态码对比

### 共同的状态码

| statusCode | 含义 | `/recognize` | `/recognize/batch` |
|-------------|------|:------------:|:------------------:|
| **200** | 识别成功且匹配到人物 | ✅ | ✅ |
| **201** | 未检测到人脸 | ✅ | ✅ |
| **251** | 数据库为空 | ✅ | ✅ |
| **252** | 未匹配到对象 | ✅ | ✅ |

### 差异点

| statusCode | 含义 | `/recognize` | `/recognize/batch` | 说明 |
|-------------|------|:------------:|:------------------:|------|
| **202** | 人脸尺寸过小 | ✅ | ❌ | 批量接口中记录在单帧的 `error` 字段 |
| **400** | 请求参数错误 | ❌ | ✅ | 批量接口用于验证 photos 列表不能为空 |
| **401** | base64 解码失败 | ✅ | ❌ | 批量接口中记录在单帧的 `error` 字段 |
| **402** | 图片格式错误 | ✅ | ❌ | 批量接口中记录在单帧的 `error` 字段 |
| **403** | 无效图片数据 | ✅ | ❌ | 批量接口中记录在单帧的 `error` 字段 |
| **501** | 人脸检测服务错误 | ✅ | (内部处理) | 批量接口中记录在单帧的 `error` 字段 |
| **502** | 特征提取失败 | ✅ | (内部处理) | 批量接口中记录在单帧的 `error` 字段 |

**为什么会有差异？**

批量识别接口 (`/recognize/batch`) 采用了**逐帧处理 + 结果聚合**的策略：
- 单帧的图片解析错误、人脸过小等问题**不会导致整个请求失败**
- 这些错误会被记录在 `data.frames[i].error` 字段中
- 只有当**所有帧都失败**或**参数验证失败**时，才会返回错误状态码
- 这种设计提高了批量接口的鲁棒性，即使部分帧有问题，仍能从有效帧中识别出结果

---

## 📝 注意事项

1. **targets 参数格式**：只接收人员编号列表（如 `["T001", "T002"]`）
2. **targets 匹配阈值**：使用 `threshold / 2` 作为候选阈值，但响应中的 `threshold` 仍是全局阈值
3. **match 列表排序**：按相似度降序排列，`match[0]` 即最相似结果
4. **similarity 格式**：百分比字符串（如 "87.45%"），便于直接展示
5. **statusCode 判断顺序**：
   - 先判断 200（成功匹配）
   - 再判断 201/202（人脸相关）
   - 再判断 251/252（数据库相关）
   - 最后判断 4xx/5xx（错误）
6. **批量接口的错误处理**：批量接口会尽可能处理所有帧，单帧错误不会中断整个请求，错误信息记录在 `frames[i].error` 中

---

## 图片要求

**支持格式**：JPG, PNG, BMP

**尺寸要求**：
- 最小人脸尺寸：根据配置文件 `face.rec_min_face_hw` 设置（默认 80x80 像素）
- 推荐图片分辨率：≥ 640x480

**质量要求**：
- 人脸清晰可见
- 光线充足
- 正面或接近正面角度
- 无遮挡（口罩、墨镜等）

---

## 更新日志

| 版本 | 日期 | 变更内容 |
|------|------|---------|
| v5.0 | 2026-01-13 | **重大变更**: 统一响应格式，所有接口 HTTP 状态码永远返回 200，通过 `statusCode` 字段区分成功/失败。新增细粒度状态码（201/202/251/252/401/402/403）。这是一个**破坏性变更**，需要客户端/前端适配。 |
| v2.1 | 2026-01-09 | 更新响应结构（match 列表 + is_target），补充 batch 接口说明 |
| v2.0 | 2026-01-06 | 统一返回 HTTP 200 + 结构化响应 |
| v1.0 | - | 初始版本 |

---

## 相关文档

- [PERSONS_API.md](./PERSONS_API.md) - 人物管理接口文档
- [DEPLOYMENT.md](./DEPLOYMENT.md) - 部署说明文档

---

## 📞 技术支持

- 邮箱：<seonzheung@gmail.com>
