# =============================================================================
# FaceRecAPI 多阶段构建 Dockerfile
#
# Stage 1 (builder): CUDA 11.8 devel + Cython 3.2.4 → 编译业务代码为 .so
# Stage 2 (runtime): CUDA 11.8 runtime + 仅装运行时依赖 + 仅 .so/.py(白名单)
#
# 构建上下文(由 docker-compose.yml 指定): app/ 目录
#   即所有 COPY 的源路径都相对于 app/
# =============================================================================

# -----------------------------------------------------------------------------
# Stage 1: builder
# -----------------------------------------------------------------------------
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# python3.10 + 编译工具(dlib、Cython 都需要)
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.10 python3.10-dev python3-pip \
        build-essential cmake \
        libboost-all-dev \
        libgl1 libglib2.0-0 \
        ca-certificates curl \
    && ln -sf /usr/bin/python3.10 /usr/bin/python \
    && ln -sf /usr/bin/python3.10 /usr/bin/python3 \
    && rm -rf /var/lib/apt/lists/*

# pip 索引源策略:
#   主源     = pypi.org   （包同步完整，当前构建环境下载更稳定）
#   备源     = 阿里云镜像（国内网络兜底）
#   wheel 源 = paddlepaddle.org.cn （fastdeploy-gpu-python 1.0.7 专属,在 requirements.txt 内 --find-links）
ENV PIP_INDEX_URL=https://pypi.org/simple \
    PIP_EXTRA_INDEX_URL=https://mirrors.aliyun.com/pypi/simple \
    PIP_TRUSTED_HOST="mirrors.aliyun.com pypi.org files.pythonhosted.org bj.bcebos.com www.paddlepaddle.org.cn" \
    PIP_DEFAULT_TIMEOUT=180

RUN pip install --upgrade pip setuptools wheel

# 预装 build 期间会被 sdist 包(insightface 0.7.3 / dlib 20.0.0)需要的依赖
# 锁定 numpy 1.26.4 防止 PEP 517 build isolation 拉 numpy 2.x（onnxruntime-gpu 1.16.3 不兼容）
# Cython 3.2.4 与 conda 环境一致
RUN pip install numpy==1.26.4 Cython==3.2.4

# 业务依赖(供 Cython 编译期 import 检查 + 运行期使用)
# --no-build-isolation: insightface/dlib 编译时复用主环境的 numpy 1.26.4 + setuptools,
#                       避免 build isolation 重复下载 numpy 2.x(慢且容易中断)
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-build-isolation -r /tmp/requirements.txt

# ------ 编译业务代码 ------
WORKDIR /build
# .dockerignore 排除 media / logs / config.toml / tests / docker / 等；
# ai_models 保留在构建上下文中，随镜像内置。
COPY . /build/app/

WORKDIR /build/app
RUN set -eux; \
    test -f ai_models/shape_predictor_68_face_landmarks.dat; \
    test -f ai_models/ms1mv3_arcface_r100.onnx; \
    test -f ai_models/models/buffalo_l/det_10g.onnx; \
    test -f ai_models/models/buffalo_l/w600k_r50.onnx; \
    test -f ai_models/models/buffalo_l/2d106det.onnx; \
    test -f ai_models/models/buffalo_l/1k3d68.onnx; \
    test -f ai_models/models/buffalo_l/genderage.onnx
RUN python build_cython.py build_ext --inplace -j 4 \
    && python clean_after_build.py \
    && rm -f build_cython.py clean_after_build.py \
    && find . -name "*.pyc" -delete \
    && find . -name "__pycache__" -type d -exec rm -rf {} +

# 此时 /build/app/ 内只剩:
#   main.py(入口,不编译) + models/(*.py,Pydantic 不编译) + 各业务目录(*.so + __init__.py)
#   以及随构建上下文复制进来的 ai_models/ 模型文件。


# -----------------------------------------------------------------------------
# Stage 2: runtime
# -----------------------------------------------------------------------------
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04 AS runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    TZ=Asia/Shanghai \
    PYTHONPATH=/srv

# 运行时只需要:
#   - python3.10
#   - libgl1 / libglib2.0-0    : opencv-python(4.11.0.86) import 时需要的最小 GUI 库
#                                 业务代码只用图像处理 API,不调用 imshow 等 GUI 函数
#   - libgomp1                 : onnxruntime 多线程
#   - libboost-thread / system : dlib 运行时
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.10 \
        libgl1 libglib2.0-0 libgomp1 \
        libboost-thread1.74.0 libboost-system1.74.0 \
        ca-certificates tzdata \
    && ln -sf /usr/bin/python3.10 /usr/bin/python \
    && ln -sf /usr/bin/python3.10 /usr/bin/python3 \
    && ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone \
    && rm -rf /var/lib/apt/lists/*

# 从 builder 复制已安装的 Python 包(不再 pip install,加速构建)
COPY --from=builder /usr/local/lib/python3.10/dist-packages /usr/local/lib/python3.10/dist-packages
COPY --from=builder /usr/local/bin/uvicorn /usr/local/bin/uvicorn
COPY --from=builder /usr/local/bin/gunicorn /usr/local/bin/gunicorn

# 复制编译后的业务代码到 /srv/app/
WORKDIR /srv
COPY --from=builder /build/app /srv/app

# 模型已随镜像复制；media/logs 仍由 volume 挂载，先创建目录作为占位
RUN mkdir -p /srv/app/media/person_photos /srv/app/logs
RUN set -eux; \
    test -f /srv/app/ai_models/shape_predictor_68_face_landmarks.dat; \
    test -f /srv/app/ai_models/ms1mv3_arcface_r100.onnx; \
    test -f /srv/app/ai_models/models/buffalo_l/det_10g.onnx; \
    test -f /srv/app/ai_models/models/buffalo_l/w600k_r50.onnx; \
    test -f /srv/app/ai_models/models/buffalo_l/2d106det.onnx; \
    test -f /srv/app/ai_models/models/buffalo_l/1k3d68.onnx; \
    test -f /srv/app/ai_models/models/buffalo_l/genderage.onnx

EXPOSE 8003

# 健康检查
HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
    CMD python -c "import urllib.request,sys;urllib.request.urlopen('http://127.0.0.1:8003/ops/health',timeout=3);" || exit 1

# 单进程 uvicorn:GPU/模型只加载一次,避免显存翻倍
CMD ["python", "-m", "uvicorn", "app.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8003", \
     "--workers", "1", \
     "--no-access-log"]
