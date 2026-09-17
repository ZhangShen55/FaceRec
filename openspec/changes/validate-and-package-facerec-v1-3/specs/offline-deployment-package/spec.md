## Purpose

规定 FaceRec v1.3 离线交付包的构建、运行验收、镜像导出、安装资料和完整性校验要求，使交付人员无需源码构建即可在符合条件的 GPU 服务器上完成部署。

## ADDED Requirements

### Requirement: 固定版本镜像构建
系统 SHALL 使用当前版本的 Dockerfile 完整构建镜像 `facerec:v1.3`，且构建过程必须在任一必需 AI 模型缺失或 Cython 编译失败时失败。

#### Scenario: 完整构建成功
- **WHEN** 构建上下文包含规定的 2 个根目录模型和 `buffalo_l` 的 5 个 ONNX 模型，并且依赖可用
- **THEN** Docker 构建成功生成仓库名和标签严格为 `facerec:v1.3` 的镜像

#### Scenario: 模型缺失阻止构建
- **WHEN** 任一必需模型未进入构建上下文或最终镜像
- **THEN** Docker 构建必须失败且不得生成可交付镜像

### Requirement: 最终镜像内容可验证
最终镜像 MUST 包含 `/srv/app/ai_models` 下的全部必需模型，并 SHALL 使用 Cython 扩展承载 `core`、`services`、`router`、`middleware` 和 `utils` 的受保护业务模块。

#### Scenario: 模型完整性检查
- **WHEN** 检查已构建的 `facerec:v1.3` 文件系统
- **THEN** 2 个根目录模型及 `models/buffalo_l/` 下的 5 个 ONNX 文件均存在且非空

#### Scenario: Cython 产物检查
- **WHEN** 检查最终镜像中的业务目录
- **THEN** 应编译模块存在对应的 `.cpython-310-*.so` 文件，且除 `__init__.py`、`core/config.py`、`models/` 和 `main.py` 白名单外不保留对应业务源码

### Requirement: 三服务运行验收
系统 SHALL 在同一 `facerec-network` 上运行 `facerec-mongo`、`facerec-redis` 和 `facerec-api`，并在导出 FaceRec 镜像前完成数据库、缓存、GPU、模型初始化和 HTTP 接口验收。

#### Scenario: 三服务健康启动
- **WHEN** 使用验证配置启动三个服务
- **THEN** MongoDB 和 Redis 健康检查通过，FaceRecAPI 容器保持运行并且 `/ops/health` 返回 HTTP 200 和 `status=healthy`

#### Scenario: 内部服务名解析
- **WHEN** FaceRecAPI 在 `facerec-network` 内连接依赖服务
- **THEN** MongoDB 使用 `facerec-mongo:27017`，Redis 使用 `facerec-redis:6379`，均不依赖宿主机映射地址

#### Scenario: GPU 和模型初始化
- **WHEN** FaceRecAPI 完成启动
- **THEN** 容器内可见所选 NVIDIA GPU，启动日志确认 InsightFace 和 FastDeploy ArcFace 已成功预加载，且不存在回退或模型缺失错误

#### Scenario: 人脸检测烟雾测试
- **WHEN** 使用仓库测试人脸图片调用 `/recognize`
- **THEN** 请求通过图片解析和人脸检测阶段，返回符合统一响应结构的业务结果且不得返回图片解析、人脸检测或模型加载错误码

### Requirement: 验证通过后导出镜像
系统 MUST 仅在三服务运行验收全部通过后导出 `facerec:v1.3`，输出文件名 SHALL 为 `faceRec_v1.3.tar`。

#### Scenario: 成功导出
- **WHEN** 所有构建和运行验收项通过
- **THEN** `docker save` 生成非空的 `deploy/facerec/faceRec_v1.3.tar`，且该归档加载后恢复镜像标签 `facerec:v1.3`

#### Scenario: 验收失败禁止导出
- **WHEN** 任一构建内容检查或运行验收失败
- **THEN** 流程停止并保留诊断信息，不得用失败或旧镜像覆盖 `faceRec_v1.3.tar`

### Requirement: 离线交付目录完整
`deploy/facerec/` SHALL 包含部署所需的全部离线镜像和操作资料，且交付专用 Compose MUST 只引用固定镜像，不得包含源码构建配置。

#### Scenario: 交付文件清单完整
- **WHEN** 对交付目录执行完整性检查
- **THEN** 至少包含 `faceRec_v1.3.tar`、`mongo_7.0.12-jammy.tar`、`redis_7-alpine.tar`、`docker-compose.yml`、`.env`、`config.toml.template`、`install.sh`、`check.sh`、`uninstall.sh`、`SHA256SUMS`、`VERSION` 和 `安装说明.md`

#### Scenario: 交付 Compose 无构建依赖
- **WHEN** 在没有项目源码的目标服务器解析交付专用 Compose
- **THEN** Compose 仅引用 `facerec:v1.3`、`mongo:7.0.12-jammy` 和 `redis:7-alpine`，且不存在 `build` 字段

### Requirement: 安装参数与网络行为明确
交付包 SHALL 通过 `.env` 和 `config.toml.template` 生成运行配置；MongoDB 和 Redis 账号策略、GPU 映射、端口绑定及容器内连接地址必须保持一致。

#### Scenario: 默认参数安装
- **WHEN** 交付人员使用默认 `.env` 执行安装
- **THEN** MongoDB 使用默认 `root/root`、Redis 不启用密码、FaceRecAPI 使用容器内逻辑 GPU 0，并通过服务名连接两个中间件

#### Scenario: MongoDB 宿主机绑定切换
- **WHEN** 将 `MONGO_BIND_ADDRESS` 设置为 `0.0.0.0` 或 `127.0.0.1` 后重新创建 `facerec-mongo`
- **THEN** 仅宿主机端口可见范围发生变化，Docker network 内的 `facerec-mongo:27017` 连接保持不变

### Requirement: 交付包可校验和可运维
交付包 MUST 提供文件完整性、安装结果和卸载行为的明确检查机制，并保护已有持久化数据。

#### Scenario: 文件校验成功
- **WHEN** 在完整且未损坏的交付目录执行 SHA-256 校验
- **THEN** `SHA256SUMS` 中列出的所有文件均通过校验

#### Scenario: 安装后检查
- **WHEN** 执行 `check.sh`
- **THEN** 脚本验证三个固定镜像、三个容器、Docker network、GPU 和 `/ops/health`，并以退出码表示成功或失败

#### Scenario: 默认卸载保留数据
- **WHEN** 执行不带清理参数的 `uninstall.sh`
- **THEN** 三个服务和网络停止或移除，但 `/data/mongo`、`/data/redis`、`/data/facerecapi` 中的数据保持不变
