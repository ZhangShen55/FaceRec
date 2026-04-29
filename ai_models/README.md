# AI 模型文件目录

本目录用于存放人脸识别系统所需的全部 AI 模型文件。系统采用 **InsightFace（检测+对齐）+ FastDeploy ArcFace（特征提取）** 的主流水线，并保留 **dlib（68 点关键点）** 作为智能回退方案。

> 引擎选择由 `app/config.toml` 中 `[face_detection]` 的 `detector` 字段控制，可选 `"insightface"` 或 `"dlib"`。详见根目录 `agent.md`。

---

## 一、目录结构（最终形态）

```
ai_models/
├── README.md                                # 本文档
├── shape_predictor_68_face_landmarks.dat    # dlib 68 点关键点模型（备用引擎）
├── ms1mv3_arcface_r100.onnx                 # FastDeploy ArcFace 特征提取模型（主特征向量）
└── models/                                  # InsightFace 标准目录（必须保留 models/ 这一级）
    └── buffalo_l/                           # InsightFace 模型套件（默认主检测/对齐引擎）
        ├── det_10g.onnx                     # 人脸检测（SCRFD-10G）
        ├── w600k_r50.onnx                   # 人脸识别（ResNet-50, MS1MV3）
        ├── 2d106det.onnx                    # 2D 106 点关键点
        ├── 1k3d68.onnx                      # 3D 68 点关键点 / 头部姿态
        └── genderage.onnx                   # 年龄性别估计
```

> **重要**：`buffalo_l` 必须放在 `ai_models/models/buffalo_l/` 下（中间多一层 `models/`），这是 `insightface.app.FaceAnalysis(name="buffalo_l", root=...)` 加载模型时强制要求的目录结构，对应 `config.toml` 中：
>
> ```toml
> [face_detection.insightface]
> model_path = "/root/workspace/FaceRecAPI_DEV/app/ai_models"
> ```
>
> 即 `model_path` 指向 `ai_models/`（而非 `ai_models/models/buffalo_l/`），InsightFace 内部会拼出完整路径。

---

## 二、必需模型清单

| 文件 | 用途 | 体积 | 引擎 | 是否必需 |
|------|------|------|------|---------|
| `shape_predictor_68_face_landmarks.dat` | dlib 68 点人脸关键点检测 | ~95 MB | dlib（备用） | 推荐保留（用于回退） |
| `ms1mv3_arcface_r100.onnx` | ArcFace 特征提取（512 维向量） | ~250 MB | FastDeploy（主） | **必需** |
| `models/buffalo_l/det_10g.onnx` | InsightFace 人脸检测（SCRFD-10G） | ~17 MB | InsightFace | **必需** |
| `models/buffalo_l/w600k_r50.onnx` | InsightFace 人脸识别 ResNet-50 | ~175 MB | InsightFace | 必需（套件完整性） |
| `models/buffalo_l/2d106det.onnx` | InsightFace 2D 106 点关键点 | ~5 MB | InsightFace | 必需（套件完整性） |
| `models/buffalo_l/1k3d68.onnx` | InsightFace 3D 68 点 / 姿态 | ~143 MB | InsightFace | 必需（套件完整性） |
| `models/buffalo_l/genderage.onnx` | 年龄性别估计 | ~1.3 MB | InsightFace | 必需（套件完整性） |

> **总体积约 690 MB**，部署前请确认磁盘空间。
>
> 当前实现里**特征向量统一由 `ms1mv3_arcface_r100.onnx` 输出**（无论检测路径走 InsightFace 还是 dlib），以保证特征空间一致性、可互比对。`w600k_r50.onnx` 仅作为 InsightFace 套件依赖加载，不参与最终的相似度计算。

---

## 三、模型获取方式

### 1. dlib 关键点模型

- 官方源：<http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2>
- 网盘备份：<https://pan.baidu.com/s/1-pr9VcAZk1Xd4LLam9c-jw?pwd=5555>
- 下载后 `bzip2 -d shape_predictor_68_face_landmarks.dat.bz2` 解压。

### 2. FastDeploy ArcFace 模型

- 网盘备份（推荐）：<https://pan.baidu.com/s/1-pr9VcAZk1Xd4LLam9c-jw?pwd=5555>
- 或从 InsightFace 官方仓库 `model_zoo` 中获取 `ms1mv3_arcface_r100` 的 ONNX 版本。

### 3. InsightFace `buffalo_l` 模型套件

**方式 A：自动下载（首次运行联网时）**

`insightface` 库会在第一次调用 `FaceAnalysis(name="buffalo_l")` 时自动从 GitHub Release 下载 `buffalo_l.zip`，默认解压到 `~/.insightface/models/buffalo_l/`。可以将该目录整体复制到本目录的 `models/` 下。

**方式 B：离线手动下载（推荐生产环境）**

- 下载 `buffalo_l.zip`：<https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip>
- 解压后获得包含上述 5 个 `.onnx` 文件的 `buffalo_l/` 目录
- 整体复制到 `ai_models/models/buffalo_l/`

**方式 C：网盘备份**

- <https://pan.baidu.com/s/1-pr9VcAZk1Xd4LLam9c-jw?pwd=5555>

---

## 四、安装步骤

```bash
cd /root/workspace/FaceRecAPI_DEV/app/ai_models

# 1) dlib 关键点模型
wget http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
bzip2 -d shape_predictor_68_face_landmarks.dat.bz2

# 2) ArcFace 特征模型（请用网盘下载到当前目录）
# 结果：ms1mv3_arcface_r100.onnx

# 3) InsightFace buffalo_l 套件
mkdir -p models
cd models
wget https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip
unzip buffalo_l.zip       # 解压后会得到 buffalo_l/ 目录
rm buffalo_l.zip

# 4) 校验目录结构
cd /root/workspace/FaceRecAPI_DEV/app/ai_models
tree -L 3
```

---

## 五、容器化部署建议

| 策略 | 优点 | 缺点 |
|------|------|------|
| **挂载方式**（推荐） | 镜像小、模型可独立升级 | 需要在宿主机维护模型目录 |
| **打包到镜像** | 部署即用、无外部依赖 | 镜像体积 +700 MB |

`docker/docker-compose.yml` 默认采用**只读挂载**：

```yaml
volumes:
  - ../ai_models:/app/ai_models:ro
```

如果选择打包到镜像，可在 `Dockerfile` 中执行 `COPY ai_models/ /app/ai_models/`。

---

## 六、备选模型（可选）

InsightFace 还提供更轻量的 `buffalo_s`（约 170 MB），适合资源受限的边缘部署。切换方式：

1. 下载 `buffalo_s.zip` 解压到 `ai_models/models/buffalo_s/`
2. 修改 `config.toml`：
   ```toml
   [face_detection.insightface]
   model_name = "buffalo_s"
   ```

---

## 七、注意事项

- **必须**严格保持文件名与目录层级一致，否则会触发：
  - `dlib`：`RuntimeError: Unable to open shape_predictor_68_face_landmarks.dat`
  - `InsightFace`：模型下载会重新触发（联网失败则启动失败）
  - `FastDeploy`：`ArcFace` 加载报 `FileNotFound`
- 若启动日志中看到 `⚠️  InsightFace 不可用，将降级到 dlib`，请检查 GPU/驱动以及 `models/buffalo_l/` 完整性。
- GPU 部署需要确保 `onnxruntime-gpu` 与系统 CUDA 版本匹配（项目默认 CUDA 11.6/11.8 + `onnxruntime-gpu==1.16.3`）。
