#!/usr/bin/env bash
# =============================================================================
# FaceRecAPI 宿主机一次性初始化脚本（适配 compose 全栈部署）
#
# 职责（幂等,可重复执行）：
#   1. 创建宿主机数据目录
#        /data/mongo                       (mongo 数据卷)
#        /data/redis                       (redis 数据卷)
#        /data/facerecapi/{ai_models,media/person_photos,logs}
#   2. 从仓库 app/config.toml 生成容器版 /data/facerecapi/config.toml
#      自动改写 5 处:
#        - [db].host                                   → mongo7
#        - [redis].host                                → redis
#        - [face_detection.insightface].model_path     → /srv/app/ai_models
#        - [face_detection.insightface].gpu_id         → 0
#        - [gpu].gpu_id                                → 0
#   3. 检查模型文件是否就位,缺失则提示
#
# 注意：
#   - 不创建 docker network、不动任何容器（compose 会自管 facerec-net）
#   - 不写入业务数据
#   - 单 GPU 容器内 gpu_id 永远为 0
#
# 用法：
#   sudo bash init_host.sh                 # 已有 config.toml 时保留不动
#   sudo bash init_host.sh --force-config  # 备份并重新生成容器版 config.toml
# =============================================================================
set -euo pipefail

# ---------- 配置 ----------
DATA_FACEREC="/data/facerecapi"
DATA_MONGO="/data/mongo"
DATA_REDIS="/data/redis"
FORCE_CONFIG=0

for arg in "$@"; do
    case "$arg" in
        --force-config)
            FORCE_CONFIG=1
            ;;
        -h|--help)
            echo "Usage: sudo bash init_host.sh [--force-config]"
            exit 0
            ;;
        *)
            error "未知参数: $arg"
            exit 1
            ;;
    esac
done

# 仓库内的模板（基于本脚本位置定位 app/）
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APP_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
TEMPLATE_CFG="${APP_DIR}/config.toml"

# ---------- 颜色输出 ----------
GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; NC='\033[0m'
info()  { echo -e "${GREEN}[INFO]${NC}  $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
error() { echo -e "${RED}[ERROR]${NC} $*" >&2; }

# ---------- 0) 前置检查 ----------
[ "$(id -u)" -eq 0 ] || { error "需要 root 权限运行（写 /data 目录）"; exit 1; }
[ -f "${TEMPLATE_CFG}" ] || { error "找不到模板 config.toml: ${TEMPLATE_CFG}"; exit 1; }

# ---------- 1) 创建宿主机目录 ----------
info "1/3 创建宿主机数据目录 ..."
mkdir -p "${DATA_MONGO}"
mkdir -p "${DATA_REDIS}"
mkdir -p "${DATA_FACEREC}/ai_models"
mkdir -p "${DATA_FACEREC}/media/person_photos"
mkdir -p "${DATA_FACEREC}/logs"
info "    /data/mongo /data/redis /data/facerecapi/{ai_models,media,logs} 已就绪"

# ---------- 2) 生成容器版 config.toml ----------
TARGET_CFG="${DATA_FACEREC}/config.toml"
if [ -f "${TARGET_CFG}" ] && [ "${FORCE_CONFIG}" -eq 0 ]; then
    warn "2/3 目标已存在: ${TARGET_CFG}（保留不动）"
    warn "    如需重新生成容器版配置，请执行: sudo bash ${0} --force-config"
else
    if [ -f "${TARGET_CFG}" ] && [ "${FORCE_CONFIG}" -eq 1 ]; then
        backup="${TARGET_CFG}.bak.$(date +%Y%m%d%H%M%S)"
        cp "${TARGET_CFG}" "${backup}"
        warn "2/3 已备份旧配置: ${backup}"
    fi
    info "2/3 生成容器版 config.toml → ${TARGET_CFG}"
    cp "${TEMPLATE_CFG}" "${TARGET_CFG}"

    # 改写策略:用 sed 精准匹配 key,中间值用 [^"]* 避免贪婪吞注释
    sed -i -E 's|^(host[[:space:]]*=[[:space:]]*")[^"]*(".*#[[:space:]]*数据库地址.*)$|\1mongo7\2|' "${TARGET_CFG}"
    sed -i -E 's|^(host[[:space:]]*=[[:space:]]*")[^"]*(".*#[[:space:]]*Redis[[:space:]]*服务器地址.*)$|\1redis\2|' "${TARGET_CFG}"
    sed -i -E 's|^(model_path[[:space:]]*=[[:space:]]*")[^"]*(".*)$|\1/srv/app/ai_models\2|' "${TARGET_CFG}"
    sed -i -E 's|^(gpu_id[[:space:]]*=[[:space:]]*)[0-9]+(.*)$|\10\2|' "${TARGET_CFG}"

    chmod 644 "${TARGET_CFG}"
    info "    已自动改写 [db].host / [redis].host / model_path / gpu_id 共 5 处"
    warn "    请确认 [db].password、[redis].password、[face].threshold 等业务参数符合预期"
fi

# ---------- 3) 检查模型文件 ----------
info "3/3 检查模型文件 ..."
MISSING=0
for f in \
    "${DATA_FACEREC}/ai_models/shape_predictor_68_face_landmarks.dat" \
    "${DATA_FACEREC}/ai_models/ms1mv3_arcface_r100.onnx" \
    "${DATA_FACEREC}/ai_models/buffalo_l/det_10g.onnx" \
    "${DATA_FACEREC}/ai_models/buffalo_l/w600k_r50.onnx"; do
    if [ ! -f "$f" ]; then
        warn "    缺少模型: $f"
        MISSING=$((MISSING+1))
    fi
done

if [ "${MISSING}" -gt 0 ]; then
    warn "存在 ${MISSING} 个模型文件缺失,请将源机的 ai_models 目录拷贝过来:"
    warn "    cp -r /path/to/source/ai_models/. ${DATA_FACEREC}/ai_models/"
    warn "    （参考 app/ai_models/README.md 获取下载链接）"
else
    info "    模型文件齐全 ✅"
fi

echo ""
info "============================================================"
info "宿主机初始化完成"
info "下一步:"
info "    cd ${APP_DIR}/docker"
info "    docker compose up -d           # 一键拉起 mongo7 + redis + facerecapi"
info "    docker compose logs -f facerecapi"
info "============================================================"
