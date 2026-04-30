# app/main.py
import os
from pathlib import Path
from contextlib import asynccontextmanager
from concurrent.futures import ProcessPoolExecutor
from contextvars import ContextVar
from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

# 在导入其他模块之前，先导入 config 来读取配置
from app.core.config import settings

# 如果 InsightFace 配置使用非 GPU 0，设置 CUDA_VISIBLE_DEVICES 环境变量
# 让主进程只看到配置的 GPU，避免在 GPU 0 上初始化资源
# 注意：这必须在导入 ai_engine 之前设置，因为导入可能会触发 GPU 初始化
if hasattr(settings, 'face_detection') and settings.face_detection.detector.lower() == 'insightface':
    if_config = settings.face_detection.insightface
    if if_config.device.lower() == 'gpu' and if_config.gpu_id != 0:
        # 如果 InsightFace 使用非 GPU 0，设置 CUDA_VISIBLE_DEVICES
        # 设置后，原来的 GPU 1 会重新映射为 GPU 0
        gpu_id = if_config.gpu_id
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
        # 注意：子进程中需要使用 GPU 0（因为已经重新映射了），所以需要调整配置
        # 但为了保持配置的一致性，我们在子进程中会读取原始配置并处理映射

from app.core.database import db
from app.core import ai_engine
from app.core.logger import get_logger
from app.core.db_init import init_database
from app.router import faces, persons, web, ops
from app.core.logger import request_id_ctx, new_request_id
from app.middleware import APIStatsMiddleware
from app.models.api_response import StatusCode, ApiResponse
from app.core.redis_client import RedisClient
from app.services.cache_service import cache_service

logger = get_logger(__name__)

BASE_DIR = Path(__file__).resolve().parent
MAX_WORKERS = settings.thread.max_workers


def _dummy_insightface_warmup():
    """ProcessPoolExecutor 预热任务：必须是模块顶层函数，才能被 pickle。"""
    return "InsightFace 已预加载"


# ============ 模型预加载函数 ============
async def _preload_ai_models():
    """
    启动时预加载 AI 模型到 GPU，避免首次请求时的延迟加载
    包括：
    1. InsightFace 检测模型（在子进程中加载）
    2. Embedding 模型（ArcFace，在主进程中加载）
    """
    import asyncio

    detector_choice = settings.face_detection.detector.lower()

    if detector_choice == "insightface":
        logger.info("🔄 预加载 InsightFace 检测模型...")
        try:
            # 在线程池中运行 InsightFace 初始化，以避免阻塞主线程
            # InsightFace 会在 _init_dlib_worker 初始化时加载
            # 这里通过向进程池提交一个空任务来触发初始化
            # 使用 loop 的 run_in_executor 来异步调用
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                ai_engine.GLOBAL_PROCESS_POOL,
                _dummy_insightface_warmup
            )
            logger.info(f"✅ {result}")
        except Exception as e:
            logger.warning(f"⚠️  InsightFace 预加载失败: {e}")

    logger.info("🔄 预加载 Embedding 模型 (ArcFace)...")
    try:
        # 在主进程中同步加载 Embedding 模型
        embedding_model = ai_engine._get_embedding_model()
        logger.info(f"✅ Embedding 模型已预加载到 GPU")
    except Exception as e:
        logger.warning(f"⚠️  Embedding 模型预加载失败: {e}")

# ---------------- 生命周期管理 (核心) ----------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    # ================= 启动 (Startup) =================
    logger.info("System Startup: Initializing resources...")

    try:
        await db.command({"ping": 1})
        logger.debug("MongoDB ping ok")
    except Exception as e:
        logger.exception("MongoDB ping failed: %s", e)

    # 1.1 数据库结构初始化（幂等创建索引,不动业务数据）
    await init_database(db)

    # 2. Redis 连接测试
    try:
        await RedisClient.ping()
        logger.info("✅ Redis 连接成功")
    except Exception as e:
        logger.error(f"❌ Redis 连接失败: {e}")
        logger.warning("⚠️  将在没有缓存的情况下运行")

    # 3. 启动时加载人员特征到 Redis
    if settings.redis.cache.enable_embedding_cache and settings.redis.cache.refresh_on_startup:
        try:
            count = await cache_service.reload_all_embeddings()
            logger.info(f"✅ 启动时已加载 {count} 个人员特征到 Redis")
        except Exception as e:
            logger.error(f"❌ 启动时加载人员特征失败: {e}")

    # 4. Dlib 进程池初始化 (注入到 ai_engine)
    # 确保 max_workers 设置合理 (建议 1 或 2，防止内存爆炸)
    # 当前 settings.thread.max_workers 建议设置为 2
    logger.info(f"正在初始化 Dlib 进程池，工作线程数: {MAX_WORKERS}...")
    pool = ProcessPoolExecutor(
        max_workers=MAX_WORKERS,
        initializer=ai_engine._init_dlib_worker
    )
    ai_engine.GLOBAL_PROCESS_POOL = pool
    logger.info("✅ Dlib 进程池初始化完成")

    # 5. 预加载 InsightFace 和 Embedding 模型到 GPU
    logger.info("🔄 预加载 AI 模型到 GPU...")
    try:
        await _preload_ai_models()
        logger.info("✅ AI 模型预加载完成")
    except Exception as e:
        logger.warning(f"⚠️  AI 模型预加载失败: {e}，系统将在首次请求时延迟加载")

    yield  # 应用运行中...

    # ================= 关闭 (Shutdown) =================
    logger.info("系统关闭: 释放资源...")

    # 6. 关闭 Redis 连接
    try:
        await RedisClient.close()
        logger.info("✅ Redis 连接已关闭")
    except Exception as e:
        logger.error(f"❌ Redis 关闭失败: {e}")

    # 7. 资源清理
    pool.shutdown(wait=True)
    ai_engine.GLOBAL_PROCESS_POOL = None
    logger.info("✅ Dlib 进程池关闭成功")


# ---------------- App 初始化 ----------------
app = FastAPI(
    title="人脸识别API系统",
    lifespan=lifespan
)

# ---------------- 全局异常处理器 ----------------
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """
    处理 Pydantic 验证错误，返回统一的 ApiResponse 格式
    HTTP 状态码永远是 200，通过 statusCode 字段区分错误
    """
    # 提取第一个错误信息
    errors = exc.errors()
    if errors:
        error = errors[0]
        # 获取字段名
        field = error.get('loc', [])[-1] if error.get('loc') else 'unknown'
        # 获取错误类型
        error_type = error.get('type', '')

        # 根据错误类型生成友好的错误信息
        if error_type == 'missing':
            message = f"缺少{field}参数"
        else:
            # 使用自定义的错误信息（来自 field_validator）
            message = error.get('msg', '参数验证失败')

        logger.error(f"[ValidationError] 参数验证失败: {message}, path: {request.url.path}")

        # 返回 JSONResponse，HTTP 状态码为 200
        return JSONResponse(
            status_code=200,
            content={
                "statusCode": StatusCode.BAD_REQUEST,
                "message": message,
                "data": None
            }
        )

    return JSONResponse(
        status_code=200,
        content={
            "statusCode": StatusCode.BAD_REQUEST,
            "message": "请求参数验证失败",
            "data": None
        }
    )

# ---------------- 注册中间件 ----------------
# 1. API 统计中间件（必须在 request_id 中间件之后）
app.add_middleware(APIStatsMiddleware)

@app.middleware("http")
async def add_request_id(request: Request, call_next):
    rid = new_request_id()
    request_id_ctx.set(rid)
    response = await call_next(request)
    # 响应头中添加 X-Request-Id（前端就不返回了）
    # response.headers["X-Request-Id"] = rid
    return response

# ---------------- 挂载静态资源 ----------------
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")
app.mount("/media", StaticFiles(directory=BASE_DIR / "media"), name="media")

# ---------------- 注册路由 ----------------
app.include_router(ops.router)
app.include_router(faces.router)
app.include_router(persons.router)
app.include_router(web.router)

# ---------------- 调试入口 ----------------
if __name__ == "__main__":
    import uvicorn

    # 开发环境调试用
    uvicorn.run(app, host="0.0.0.0", port=8003, reload=False)
    # PYTHONPATH=/root/workspace/FaceRecAPI_DEV OMP_NUM_THREADS=1 uvicorn app.main:app --host 0.0.0.0 --port 8003 --workers 4
    # --env-file .env 读取 .env 文件中的环境变量(该项目中为使用)

    # cd app
    # PYTHONPATH=/root/workspace/FaceRecAPI_DEV OMP_NUM_THREADS=1 uvicorn app.main:app --host 0.0.0.0 --port 8003   --workers 1
    
    # ============ 端口检查 ============
    # sudo ss -lntp | grep ':8090' 

    # ============ 后台启动命令 ============
    # conda activate facerecapi
    # cd /root/workspace/FaceRecAPI_DEV/app
    # nohup env PYTHONPATH=/root/workspace/FaceRecAPI_DEV OMP_NUM_THREADS=1 uvicorn app.main:app --host 0.0.0.0 --port 8003  --workers 1 > /root/workspace/FaceRecAPI_DEV/app/logs/facerec_server_uvicorn.log 2>&1 & echo $! > /root/workspace/FaceRecAPI_DEV/app/logs/facerec_server_uvicorn.pid

    # ============ 查看运行状态 ============
    # ps aux | grep "facerec_server_uvicorn app.main:app"
    # tail -f /root/workspace/FaceRecAPI_DEV/app/logs/facerec_server_uvicorn.log

    # ============ 关闭服务 ============
    # 方法1: 使用 PID 文件关闭（推荐，带进程检查）
    # PID_FILE=/root/workspace/FaceRecAPI_DEV/app/logs/facerec_server_uvicorn.pid
    # if [ -f $PID_FILE ]; then
    #     PID=$(cat $PID_FILE)
    #     if ps -p $PID > /dev/null 2>&1; then
    #         kill $PID && echo "进程 $PID 已终止"
    #     else
    #         echo "进程 $PID 不存在，可能已经停止"
    #     fi
    #     rm $PID_FILE
    # else
    #     echo "PID 文件不存在"
    # fi

    # 方法2: 简单关闭（不检查进程是否存在）
    # kill $(cat /root/workspace/FaceRecAPI_DEV/app/logs/facerec_server_uvicorn.pid) 2>/dev/null
    # rm /root/workspace/FaceRecAPI_DEV/app/logs/facerec_server_uvicorn.pid 2>/dev/null

    # 方法3: 查找进程并关闭
    # ps aux | grep "facerec_server_uvicorn app.main:app" | grep -v grep | awk '{print $2}' | xargs kill

    # 方法4: 强制关闭 (慎用)
    # ps aux | grep "facerec_server_uvicorn app.main:app" | grep -v grep | awk '{print $2}' | xargs kill -9