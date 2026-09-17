# FaceRec 模型目录

本目录保存运行时模型。模型二进制由部署资产注入并受 Git 忽略，仓库只跟踪本文档。

```text
ai_models/
├── README.md
├── ms1mv3_arcface_r100.onnx
├── shape_predictor_68_face_landmarks.dat
└── models/buffalo_l/
    ├── det_10g.onnx
    ├── w600k_r50.onnx
    ├── 2d106det.onnx
    ├── 1k3d68.onnx
    └── genderage.onnx
```

七个文件都必须存在且非空。`face_detection.insightface.model_path` 指向项目根相对目录
`ai_models`；InsightFace 会在其下查找 `models/buffalo_l/`。运行时不会联网下载模型，也不
会在模型不完整、损坏或 CUDA provider 不可用时静默切换检测器或 CPU。

`buffalo_l` 负责检测与五点对齐，最终 512 维 embedding 始终由
`ms1mv3_arcface_r100.onnx` 生成。`shape_predictor_68_face_landmarks.dat` 供显式配置的
Dlib 兼容检测路径使用。

注入模型后应逐文件核对非零大小和可信来源 SHA-256，再执行：

```bash
conda run -n facerecapi python -m pytest -q \
  tests/test_insightface_worker.py tests/test_dlib_worker_startup.py
```

Dockerfile 会把 `ai_models/` 复制进镜像；独立 Compose 也可用只读挂载覆盖该目录。模型
文件、外部 manifest、部署 tar、人脸图片和凭据不得提交 Git。
