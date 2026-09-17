## Why

当前 `v1.3` 已调整 Dockerfile、模型内置方式和三服务网络配置，但尚未经过一次完整、可复现的镜像构建与运行验收。交付人员还缺少一个无需源码构建、包含固定镜像与安装资料、可校验完整性的离线部署目录。

## What Changes

- 使用当前 `docker/Dockerfile` 完整构建固定镜像 `facerec:v1.3`，验证模型文件被打入最终镜像且 Cython 编译产物符合预期。
- 使用 `facerec-mongo`、`facerec-redis`、`facerec-api` 和 `facerec-network` 启动三服务，验证容器健康、内部 DNS、数据库/缓存连接、GPU 可见性和 FaceRecAPI 健康接口。
- 在验证通过后将 `facerec:v1.3` 导出为 `faceRec_v1.3.tar`。
- 整理 `deploy/facerec/` 为可直接交付的离线安装包，包含三个镜像 tar、交付专用 Compose、`.env`、配置模板、安装/检查/卸载脚本、版本信息、SHA-256 校验清单和中文安装说明。
- 交付专用 Compose 只引用固定镜像，不包含 `build` 配置；FaceRecAPI 通过 Docker 服务名访问 MongoDB 和 Redis，MongoDB 同时发布可配置的宿主机端口，Redis 不发布宿主机端口。
- 保留已有 MongoDB/Redis tar，统一并核对交付文件名、镜像仓库名和标签。

## Capabilities

### New Capabilities

- `offline-deployment-package`: 定义 FaceRec v1.3 镜像构建验证、三服务运行验收、离线镜像导出及交付包完整性要求。

### Modified Capabilities

无。

## Impact

- 构建与运行：`docker/Dockerfile`、`docker/docker-compose.yml`、Docker/Compose、NVIDIA Container Toolkit 和 GPU 运行环境。
- 交付产物：`deploy/facerec/` 下的镜像 tar、配置模板、编排文件、脚本、校验清单和中文文档。
- 本地运行状态：apply 阶段会构建约数 GB 的镜像，并启动/重建 `facerec-mongo`、`facerec-redis`、`facerec-api` 容器；持久化数据使用 `/data/mongo`、`/data/redis` 和 `/data/facerecapi`。
- 版本与兼容性：固定 FaceRec 镜像为 `facerec:v1.3`，MongoDB 为 `mongo:7.0.12-jammy`，Redis 为 `redis:7-alpine`。
