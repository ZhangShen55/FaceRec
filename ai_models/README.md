# AI 模型文件目录

此目录用于存放人脸识别所需的AI模型文件。

## 必需的模型文件

### 1. shape_predictor_68_face_landmarks.dat
- **用途**: Dlib 68点人脸关键点检测模型
- **下载**: [http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2](http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2)
- **大小**: ~95 MB (解压后)

### 2. ms1mv3_arcface_r100.onnx
- **用途**: ArcFace 人脸特征提取模型 (ONNX格式)
- **下载**: 从官方渠道获取或联系项目维护者
- **大小**: ~250 MB

## 安装说明

1. 下载上述两个模型文件
2. 解压 shape_predictor_68_face_landmarks.dat.bz2 (如果是压缩包)
3. 将两个文件放置到本目录
4. 确保文件名与上述一致

## 文件结构

```
ai_models/
├── README.md
├── shape_predictor_68_face_landmarks.dat
└── ms1mv3_arcface_r100.onnx
```

## 注意事项

- 模型文件较大，不建议提交到 Git 仓库
- 可以使用 Git LFS 管理大文件
- 或者在 CI/CD 流程中自动下载
