# FaceRecAPI Docker 部署指南

> Cython 加密业务代码 + 多阶段构建 → 运行时镜像内核心逻辑以 `.so` 形式存在（仅 `main.py`、`models/`、`core/config.py` 这类入口/数据模型/配置模型保留）。
> 模型、配置、日志、媒体文件全部 volume 挂载，**镜像内零业务数据**。
> Compose 一键编排 `mongo7 + redis + facerecapi`，目标服务器**无需任何前置容器**。

---

## 一、服务拓扑

```
┌────────────────────── facerec-net (compose 自管 bridge 网络) ──────────────────┐
│                                                                                │
│   facerecapi (本仓库构建)        mongo7 (mongo:7.0.12-jammy)                  │
│   ─ 8003  HTTP API               ─ 27017 MongoDB                              │
│   ─ GPU x1                       ─ root/root                                  │
│   ─ /srv/app                     redis  (redis:7-alpine)                      │
│                                  ─ 6379  Redis (AOF 持久化)                   │
│                                                                                │
└────────────────────────────────────────────────────────────────────────────────┘
       │
       ├── /data/mongo       ← MongoDB 数据
       ├── /data/redis       ← Redis 数据
       └── /data/facerecapi  ← 配置 / 模型 / 媒体 / 日志
```

容器之间用容器名作为 hostname 通信（compose 自动 DNS）：
- `facerecapi → mongo7:27017`
- `facerecapi → redis:6379`

---

## 二、宿主机目录布局

```
/data/
├── mongo/                       # mongo:7 数据卷
├── redis/                       # redis:7 数据卷（AOF）
└── facerecapi/
    ├── config.toml              # 容器版配置（init_host.sh 自动生成）
    ├── ai_models/               # 模型文件（手动拷贝）
    │   ├── shape_predictor_68_face_landmarks.dat
    │   ├── ms1mv3_arcface_r100.onnx
    │   └── buffalo_l/
    │       ├── det_10g.onnx
    │       ├── 2d106det.onnx
    │       ├── 1k3d68.onnx
    │       ├── genderage.onnx
    │       └── w600k_r50.onnx
    ├── media/
    │   └── person_photos/       # is_persistence=false 时为空
    └── logs/
        └── facere_service.log
```

容器内挂载点：

| 宿主机 | 容器内 | 挂载方 | 模式 |
|---|---|---|---|
| `/data/mongo` | `/data/db` | mongo7 | rw |
| `/data/redis` | `/data` | redis | rw |
| `/data/facerecapi/config.toml` | `/srv/app/config.toml` | facerecapi | ro |
| `/data/facerecapi/ai_models` | `/srv/app/ai_models` | facerecapi | ro |
| `/data/facerecapi/media` | `/srv/app/media` | facerecapi | rw |
| `/data/facerecapi/logs` | `/srv/app/logs` | facerecapi | rw |

---

## 三、部署流程（核心 5 步）

### 阶段 A：开发机打包镜像（一次性）

```bash
# 1. 拉公共镜像
docker pull mongo:7.0.12-jammy
docker pull redis:7-alpine

# 2. 构建 facerecapi 镜像（多阶段 Cython 编译）
cd /root/workspace/FaceRecAPI_DEV/app/docker
docker compose build facerecapi

## 2.1 指定构建image名和tag名
FACERECAPI_IMAGE=jy-algorithm-app-facerec-service \
FACERECAPI_TAG=v1.0.0_0430 \
docker compose build facerecapi 


# 首次约 8-15 分钟（取决于网络与 CPU 核数）

# 3. 导出 3 个 tar 包
mkdir -p /tmp/facerec-dist
docker save -o /tmp/facerec-dist/mongo-7.0.12-jammy.tar  mongo:7.0.12-jammy
docker save -o /tmp/facerec-dist/redis-7-alpine.tar      redis:7-alpine
docker save -o /tmp/facerec-dist/facerecapi-latest.tar   facerecapi:latest

# 同时把仓库 app/ 目录(至少 docker/、scripts/、config.toml、ai_models/) 一起打包
tar czf /tmp/facerec-dist/facerec-app.tar.gz -C /root/workspace/FaceRecAPI_DEV app
```

> 镜像总大小预估：mongo ≈ 800MB / redis ≈ 40MB / facerecapi ≈ 5GB（CUDA 11.8 runtime + 推理依赖）。

### 阶段 B：传输到目标服务器

```bash
scp /tmp/facerec-dist/* user@target-server:/tmp/
# 或 rsync / U 盘 / 内网仓库,按实际网络环境
```

### 阶段 C：目标服务器一键部署

```bash
# 1) 导入 3 个镜像
cd /tmp
docker load -i mongo-7.0.12-jammy.tar
docker load -i redis-7-alpine.tar
docker load -i facerecapi-latest.tar
docker images | grep -E 'mongo|redis|facerecapi'      # 验证导入成功

# 2) 解压源码包(只用到 docker/ scripts/ config.toml ai_models/)
sudo mkdir -p /opt
sudo tar xzf facerec-app.tar.gz -C /opt
ls /opt/app                                            # 应有 docker/ scripts/ config.toml 等

# 3) 初始化宿主机目录与配置(幂等)
sudo bash /opt/app/scripts/init_host.sh
# 如目标机已有旧版 /data/facerecapi/config.toml，需要强制重新生成容器版配置：
# sudo bash /opt/app/scripts/init_host.sh --force-config

# 4) 拷贝模型文件(从源机或重新下载)
sudo cp -r /opt/app/ai_models/. /data/facerecapi/ai_models/
# 或 rsync -av /old/ai_models/ /data/facerecapi/ai_models/

# 5) (可选) 检查/调整配置
sudo vim /data/facerecapi/config.toml
# 重点确认:
#   [db].password    与 mongo7 启动密码一致(默认 root/root)
#   [db].host        compose 内部部署应为 mongo7
#   [redis].host     compose 内部部署应为 redis
########### model_path 改成挂载路径 重要 重要 重要 ###########
#   model_path       应为 /srv/app/ai_models   
########### model_path 改成挂载路径 重要 重要 重要 ###########
#   gpu_id           容器内单卡应为 0（[face_detection.insightface] 与 [gpu] 两处）
#   [redis].password 一般留空
#   [face].threshold 业务阈值
#   [media].is_persistence  推荐 false(纯后端 API)

# 6) 一键拉起
cd /opt/app/docker
docker compose up -d
docker compose ps                                      # 三个服务都应为 healthy/running
docker compose logs -f facerecapi                      # 关注启动日志
```

### 阶段 D：验证

```bash
# 健康检查
curl -s http://127.0.0.1:8003/ops/health | python3 -m json.tool

# 预期输出包含:
# "status": "healthy"
# "components": { "mongodb": "up", "redis": "up", "ai_engine": "up" }

# 接口文档: 浏览器访问 http://<服务器IP>:8003/docs
```

启动顺序由 compose `depends_on` + healthcheck 自动保证：
1. `mongo7` 启动 → `db.adminCommand('ping')` healthy
2. `redis` 启动 → `redis-cli ping` healthy
3. 上述两个 healthy 后 → `facerecapi` 启动 → 加载模型 → `/ops/health` healthy

---

## 四、运维常用命令

```bash
# 三服务总览
docker compose ps

# 查看 GPU 占用
docker exec facerecapi nvidia-smi

# 检查容器是否真的申请到 GPU
docker inspect facerecapi --format '{{json .HostConfig.DeviceRequests}}'

# 重启单服务（不影响其他）
docker compose restart facerecapi

# 完全停止（数据保留在 /data/*）
docker compose down

# 完全销毁（含网络，但数据卷仍在 /data/*）
docker compose down -v   # 注：-v 不会删除 bind mount,只会删 named volume,这里我们没用 named volume

# 升级 facerecapi（新镜像导入后）
docker load -i facerecapi-new.tar
docker compose up -d facerecapi   # 只重建该服务,mongo/redis 不动 docker-compose.yml文件中 facerecapi.image 根据实际情况修改

# 实时日志
docker compose logs -f --tail=200 facerecapi
tail -f /data/facerecapi/logs/facere_service.log

# 进入容器排查（容器内已无 .py 源码,只剩 .so）
docker exec -it facerecapi bash
```

---

## 五、首次启动正常的标志

```text
mongo7    | {"t":...,"msg":"Waiting for connections","attr":{"port":27017}}
redis     | * Ready to accept connections tcp
facerecapi| [INFO] MongoDB ping ok: ...
facerecapi| [INFO] [DBInit] persons.number 唯一索引已就绪
facerecapi| [INFO] [DBInit] persons.name 索引已就绪
facerecapi| [INFO] Redis 连接成功
facerecapi| [INFO] InsightFace 模型加载完成 ...
facerecapi| [INFO] Application startup complete.
facerecapi| [INFO] Uvicorn running on http://0.0.0.0:8003
```

---

## 六、关键风险与注意事项

1. **GPU 驱动**
   - 目标服务器需安装 NVIDIA 驱动 ≥ 520（向下兼容 CUDA 11.8）+ `nvidia-container-toolkit`
   - 验证：`docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi`
   - 若 `docker compose up` 报 `unknown or invalid runtime name: nvidia`，说明 compose 文件里使用了旧写法 `runtime: nvidia`，请删除该行并使用本文件中的 `deploy.resources.reservations.devices` 写法
   - 若报 `could not select device driver "nvidia"`，说明宿主机未安装或未配置 `nvidia-container-toolkit`

2. **端口占用**
   - 27017 / 6379 / 8003 三个端口必须空闲
   - 冲突时改 `docker-compose.yml` 中 `ports` 左侧（外部端口）

3. **mongo7 密码**
   - compose 中 `MONGO_INITDB_ROOT_PASSWORD=root` 与 `config.toml` 的 `[db].password` 必须一致
   - 改密码同步改两处

4. **数据安全**
   - 业务数据全部在 `/data/{mongo,redis,facerecapi}`，**容器删除/重建不影响数据**
   - 重要数据请按宿主机磁盘备份策略定期快照

5. **存储泄漏**
   - 默认 `[media].is_persistence = false`，`/data/facerecapi/media/person_photos` 常态为空
   - 若开启（true），需自行实现 update/delete 时的旧图清理（当前代码暂未实现）

6. **当前开发机已有 mongo7/redis 容器**
   - 在**当前开发机**直接 `docker compose up -d` 会因端口/容器名冲突失败
   - 测试方案：`docker stop mongo7 redis && docker rm mongo7 redis`（数据卷 `/data/mongo` `/data/redis` 不动），再 `docker compose up -d` 即可重新挂起，数据无损

---

## 七、加密强度说明

- 业务代码（除 `core/config.py` 外的 `core/services/router/middleware/utils`）已编译为 `.cpython-310-x86_64-linux-gnu.so`，对应源码（`.py`/`.c`）在镜像内已删除
- 反编译 `.so` 需要专用工具且产出难以阅读，对一般场景足以"反白嫖"；如需要更高强度，可叠加 PyArmor / 商业方案
- **保留 .py 不编译**的部分仅有：
  - `main.py`：uvicorn 反射启动入口
  - `models/`：Pydantic v2 数据模型，对反射敏感
  - `core/config.py`：Pydantic v2 配置模型，包含 `BaseModel` 与 `computed_field`，编译后会影响反射
  - 所有 `__init__.py`：保留包结构

---

## 八、本次新增/重写的文件清单

| 路径 | 用途 |
|---|---|
| `app/build_cython.py` | Cython 编译脚本（builder 阶段执行） |
| `app/clean_after_build.py` | 编译后清理 `.py`/`.c` |
| `app/.dockerignore` | 排除模型/媒体/日志/配置等大文件 |
| `app/docker/Dockerfile` | 多阶段构建（cuda devel→Cython→cuda runtime） |
| `app/docker/docker-compose.yml` | 三服务编排（mongo7+redis+facerecapi） |
| `app/scripts/init_host.sh` | 宿主机一次性初始化（目录/配置/模型检查） |
| `app/docker/DEPLOY.md` | 本文件 |
| `app/core/db_init.py` | 启动期幂等创建 MongoDB 索引（不动业务数据） |
