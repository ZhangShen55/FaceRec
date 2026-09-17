## 1. 构建前隔离与配置收敛

- [x] 1.1 检查 Docker、Compose、NVIDIA runtime、GPU、磁盘空间、现有同名容器和端口占用，并将结果记录到验证日志；确认不会操作来源不明的容器或既有 `/data` 数据
- [x] 1.2 更新 `.dockerignore` 排除 `deploy/`、`.agents/`、`openspec/` 等非运行时目录，同时保留 `ai_models/`，并验证构建上下文不再包含已有镜像 tar
- [x] 1.3 参数化 Compose 的 MongoDB、Redis、配置、媒体和日志挂载源，使默认值保持 `/data/*`，验证时可切换到独立临时目录；运行 `docker compose config` 验证默认值和覆盖值
- [x] 1.4 生成隔离验证配置，确认数据库主机为 `facerec-mongo`、Redis 主机为 `facerec-redis`、模型路径为 `/srv/app/ai_models`、容器内 GPU ID 为 0

## 2. FaceRec 镜像完整构建与静态验收

- [x] 2.1 使用当前 Dockerfile 完整构建 `facerec:v1.3` 并保存完整构建日志，验证模型前置检查和 Cython 编译步骤均成功且镜像标签准确
- [x] 2.2 在最终镜像中逐一检查 7 个模型文件存在且非空，并验证模型目录未被运行时 volume 覆盖
- [x] 2.3 检查 `core`、`services`、`router`、`middleware`、`utils` 中的 `.cpython-310-*.so` 产物和源码白名单，确认应保护的 `.py`/`.c` 未进入最终镜像
- [x] 2.4 使用 `docker image inspect` 记录镜像 ID、创建时间、架构和大小，确认最终镜像入口为单进程 Uvicorn 且端口为 8003

## 3. 三服务端到端运行验证

- [x] 3.1 使用临时数据目录和非冲突宿主机端口启动 `facerec-mongo`、`facerec-redis`、`facerec-api`，验证三个容器加入 `facerec-network` 且 MongoDB/Redis 健康检查通过
- [x] 3.2 从 FaceRecAPI 容器验证 `facerec-mongo:27017` 和 `facerec-redis:6379` 的 DNS 与连接，确认连接不依赖宿主机映射端口
- [x] 3.3 验证容器内 NVIDIA GPU 可见，并检查启动日志确认 InsightFace `buffalo_l` 与 FastDeploy ArcFace 均成功预加载且未触发 dlib/CPU 降级
- [x] 3.4 请求 `/ops/health` 并验证 HTTP 200、`status=healthy` 和 MongoDB 状态正常
- [x] 3.5 将仓库测试人脸图片转换为 Base64 调用 `/recognize`，验证响应符合 `{statusCode,message,data}`，且不返回 401/402/403/501/502 等解析、检测或模型错误
- [x] 3.6 汇总容器状态、健康结果、网络检查、GPU 信息、模型日志和烟雾请求结果；只有全部通过才允许继续导出镜像

## 4. 镜像导出与交付目录生成

- [x] 4.1 将 `facerec:v1.3` 先导出到临时文件，校验 tar manifest 包含 `facerec:v1.3` 后原子移动为 `deploy/facerec/faceRec_v1.3.tar`
- [x] 4.2 校验现有 MongoDB 和 Redis tar 的 manifest 分别包含 `mongo:7.0.12-jammy` 和 `redis:7-alpine`，并将文件名统一为 `mongo_7.0.12-jammy.tar` 与 `redis_7-alpine.tar`
- [x] 4.3 创建交付专用 `docker-compose.yml`，仅引用三个固定镜像且不含 `build`；运行 `docker compose config` 验证服务名、网络、GPU、端口和挂载配置
- [x] 4.4 创建 `.env` 与 `config.toml.template`，包含 API 端口、物理 GPU、MongoDB 绑定地址/端口和默认凭据，并验证模板生成的 TOML 可被项目配置模型加载
- [x] 4.5 创建幂等的 `install.sh`，验证其按顺序执行环境检查、SHA-256 校验、三个镜像加载、目录/配置生成、服务启动和健康等待，并保留已有 `/data/mongo` 配置行为
- [x] 4.6 创建 `check.sh` 和默认保留数据的 `uninstall.sh`，通过 Shell 语法检查并验证退出码能准确表达成功和失败
- [x] 4.7 创建 `VERSION` 和中文 `安装说明.md`，记录 v1.3、Git commit、镜像标签、前置环境、安装/检查/卸载步骤、MongoDB 暴露范围切换和常见故障处理

## 5. 交付包最终验收

- [x] 5.1 从 `deploy/facerec/` 的交付 Compose 和临时数据目录执行一次无源码安装验证，确认 `install.sh` 可加载镜像并启动健康的三服务
- [x] 5.2 执行交付目录的 `check.sh` 验证镜像、容器、网络、GPU 和 API，再执行 `uninstall.sh` 验证服务停止且临时持久化数据未删除
- [x] 5.3 为除 `SHA256SUMS` 自身外的所有交付文件生成校验清单并执行 `sha256sum -c SHA256SUMS`，确认全部文件校验通过
- [x] 5.4 检查交付目录最终只包含约定文件且不存在源码、模型散文件、临时日志或半成品 tar，并记录最终文件大小和 SHA-256
- [x] 5.5 停止验证容器并清理本次创建的临时数据和临时文件，确认既有业务数据和非本任务容器未被修改
