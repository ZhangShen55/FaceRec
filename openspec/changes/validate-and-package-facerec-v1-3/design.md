## Context

参见 `proposal.md` 的动机说明。当前 Dockerfile 已采用 CUDA 11.8 双阶段构建，在 builder 中执行 Cython 编译并把 `ai_models/` 复制到 runtime；Compose 已定义三个带 `facerec-` 前缀的服务和固定网络。当前机器具备 Docker 26.1.4、Compose 2.27.1 和可用的 RTX 4090，可进行完整 GPU 联调。

当前约束和已发现问题：

- `.dockerignore` 没有排除 `deploy/`、`.agents/` 和 `openspec/`。由于 `deploy/` 已包含约 790MB 镜像 tar，直接构建会显著增大上下文，并可能通过 `COPY .` 把交付包递归带入镜像。
- 当前开发 Compose 同时包含 `build` 和运行配置，不能原样交给只有离线镜像、没有源码的交付人员。
- `/ops/health` 当前检查 MongoDB 和存储，不检查 Redis 或 AI 模型，因此验收不能只依赖单个健康接口。
- 现有交付目录只有 MongoDB 和 Redis tar，且 Redis 文件名尚未统一为约定名称。
- MongoDB、Redis 和媒体目录当前固定为 `/data/*`，验证过程需要避免破坏机器上可能存在的业务数据。

## Goals / Non-Goals

**Goals:**

- 对当前 Dockerfile 做一次从依赖安装、Cython 编译到 runtime 镜像的完整构建验证。
- 使用隔离数据目录完成 MongoDB、Redis、GPU、模型和 API 的端到端运行验收。
- 生成固定标签的 FaceRec 镜像归档，并形成自包含、可校验、面向交付人员的离线部署目录。
- 让交付专用配置保持简单：固定镜像版本，环境变量只承载端口、GPU 和安装环境参数。

**Non-Goals:**

- 不改变人脸识别 API 的业务行为、匹配算法或 MongoDB 数据结构。
- 不把 Docker Engine、NVIDIA 驱动或 NVIDIA Container Toolkit 的离线安装包纳入本次交付包。
- 不验证其他 GPU 型号、CPU-only 模式或多节点部署。
- 不把 `build_cython.py` 的目录重组纳入本变更；该问题另行设计，避免干扰本次可交付性验证。

## Decisions

### 1. 在构建前收紧 Docker 上下文

向 `.dockerignore` 增加 `deploy/`、`.agents/`、`openspec/` 及其他非运行时目录。模型目录继续明确保留，因为模型必须随镜像交付。

选择该方案是因为 Dockerfile 使用 `COPY . /build/app/`，仅依赖后续清理不能阻止大型交付 tar 进入前置镜像层。替代方案是逐目录 `COPY` 白名单，隔离性更强，但本次改动范围更大，可在后续构建重构中考虑。

### 2. 构建与交付 Compose 分离

仓库的开发 Compose 可保留 `build` 以支持本地构建；`deploy/facerec/docker-compose.yml` 必须是独立的运行编排，只引用三个固定镜像。这样交付人员无需源码和构建工具。

交付服务名固定为 `facerec-api`、`facerec-mongo`、`facerec-redis`，网络固定为 `facerec-network`。FaceRecAPI 的配置模板固定使用服务名连接中间件。

### 3. 使用隔离目录和可覆盖端口进行联调

验证前将 Compose 数据挂载源参数化，默认交付仍使用 `/data/*`，验证运行则使用临时目录，例如 `/tmp/facerec-v1.3-validation/*`，并使用非冲突宿主机端口。验证结束执行 `docker compose down` 并只清理临时验证数据。

选择隔离目录而不是直接复用 `/data/*`，是为了避免测试写入或索引初始化影响已有部署。固定容器名意味着测试前必须确认同名容器不存在或仅移除本变更创建的容器。

### 4. 分层验收而非只看容器状态

验收分为四层：

1. 镜像静态检查：标签、模型文件、Cython `.so`、源码白名单。
2. 基础设施检查：MongoDB `ping`、Redis `PING`、同网络 DNS。
3. AI 运行检查：容器内 GPU 可见，启动日志确认 InsightFace 与 ArcFace 预加载。
4. API 检查：`/ops/health` 为 healthy，并使用测试图片请求 `/recognize`，确认至少通过图片解析和检测阶段。

这是因为当前健康接口不覆盖 Redis 和 AI 模型。只检查 `/ops/health` 会产生假阳性。

### 5. 验收成功后再原子替换 FaceRec tar

先将 `docker save` 输出到临时文件，校验归档非空且 manifest 包含 `facerec:v1.3`，再移动为 `deploy/facerec/faceRec_v1.3.tar`。失败时不覆盖已有交付文件。

MongoDB 和 Redis tar 读取 manifest 验证镜像标签；若文件名不规范则统一为 `mongo_7.0.12-jammy.tar` 和 `redis_7-alpine.tar`。不重新构建公共镜像。

### 6. 安装脚本从模板生成配置

`.env` 保存 `FACERECAPI_PORT`、`GPU_DEVICE_ID`、`MONGO_BIND_ADDRESS`、`MONGO_HOST_PORT` 和默认 MongoDB 凭据。`install.sh` 校验变量与 SHA-256，加载三个镜像，从 `config.toml.template` 生成 `/data/facerecapi/config.toml`，再启动 Compose。

应用配置加载器不增加环境变量展开逻辑。Redis 保持无密码；容器只映射一张物理 GPU 时，应用配置中的两个 `gpu_id` 固定为容器逻辑编号 0。

### 7. 校验清单不包含自身

`SHA256SUMS` 覆盖三个 tar、Compose、环境模板、配置模板、脚本、版本文件和安装说明，但不包含 `SHA256SUMS` 自身。生成后立即执行 `sha256sum -c SHA256SUMS` 作为交付前验收。

## Risks / Trade-offs

- [完整无缓存构建耗时长且依赖外部软件源] → 记录完整构建日志；失败时保留最后错误，不使用旧镜像继续打包。
- [模型和 CUDA 依赖导致镜像及 tar 很大] → 构建前检查磁盘空间，使用临时文件原子替换，生成 SHA-256 防止传输损坏。
- [MongoDB 默认 `0.0.0.0:27017` 且使用 `root/root`] → 安装说明明确只应在受控网络使用，可将 `MONGO_BIND_ADDRESS` 改为 `127.0.0.1` 并重建 MongoDB 容器。
- [测试容器与已有容器或端口冲突] → 使用固定前缀检测所有权、可覆盖宿主机端口和隔离数据目录；不删除无法确认来源的容器或数据。
- [仅靠日志判断模型预加载可能受日志文本变化影响] → 同时检查容器存活、GPU、模型文件和识别烟雾请求。
- [Git LFS 中的大型 tar 更新会产生较大上传成本] → apply 阶段只在验收通过后替换 FaceRec tar，并在提交前明确显示 LFS 文件变化。

## Migration Plan

1. 排除非运行时 Docker 构建上下文并参数化验证所需的数据目录。
2. 完整构建并静态检查 `facerec:v1.3`。
3. 使用隔离数据启动三服务并完成分层验收；失败则收集日志并停止，不导出镜像。
4. 验收通过后导出 FaceRec 镜像，核对三个 tar 的镜像标签和文件名。
5. 生成交付专用配置、脚本、文档、版本和 SHA-256 清单，并执行一次从交付目录出发的解析与检查。
6. 清理临时验证容器和临时数据，保留构建镜像及交付产物。

回滚时删除新生成的交付辅助文件并恢复旧 tar；镜像可通过 `docker image rm facerec:v1.3` 删除。持久化业务目录不在自动回滚范围内，也不得由默认卸载流程删除。
