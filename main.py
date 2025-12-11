# app/main.py
import os
from pathlib import Path
from contextlib import asynccontextmanager
from concurrent.futures import ProcessPoolExecutor
from contextvars import ContextVar
from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles

from app.core.database import db
from app.core import ai_engine
from app.core.config import settings
from app.core.logger import get_logger
from app.router import faces, persons, web
from app.core.logger import request_id_ctx, new_request_id

logger = get_logger(__name__)

BASE_DIR = Path(__file__).resolve().parent
MAX_WORKERS = settings.thread.max_workers

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

    # 2. Dlib 进程池初始化 (注入到 ai_engine)
    # 确保 max_workers 设置合理 (建议 1 或 2，防止内存爆炸)
    # 当前 settings.thread.max_workers 建议设置为 2
    logger.info(f"Initializing Dlib Process Pool with {MAX_WORKERS} workers...")
    pool = ProcessPoolExecutor(
        max_workers=MAX_WORKERS,
        initializer=ai_engine._init_dlib_worker
    )
    ai_engine.GLOBAL_PROCESS_POOL = pool

    yield  # 应用运行中...

    # ================= 关闭 (Shutdown) =================
    logger.info("系统关闭: 释放资源...")

    # 3. 资源清理
    pool.shutdown(wait=True)
    ai_engine.GLOBAL_PROCESS_POOL = None
    logger.info("Dlib 进程池关闭成功.")


# ---------------- App 初始化 ----------------
app = FastAPI(
    title="人脸识别API系统",
    lifespan=lifespan
)

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
app.include_router(faces.router)
app.include_router(persons.router)
app.include_router(web.router)

# ---------------- 调试入口 ----------------
if __name__ == "__main__":
    import uvicorn

    # 开发环境调试用
    uvicorn.run(app, host="0.0.0.0", port=8003, reload=False)
    # PYTHONPATH=/root/workspace/FaceRecAPI_DEV OMP_NUM_THREADS=1 uvicorn app.main:app --host 0.0.0.0 --port 8003 --workers 4 --env-file .env


    # cd app
    # PYTHONPATH=/root/workspace/FaceRecAPI_DEV OMP_NUM_THREADS=1 uvicorn app.main:app --host 0.0.0.0 --port 8003  --env-file .env --workers 1