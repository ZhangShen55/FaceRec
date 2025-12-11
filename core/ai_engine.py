# app/ai_engine.py
from typing import Optional, Tuple, List
from pathlib import Path
from bson.binary import Binary
import asyncio
import os
import cv2
import dlib
import numpy as np
import fastdeploy as fd
from concurrent.futures import ProcessPoolExecutor
from app.core.config import settings
from app.core.logger import get_logger

logger = get_logger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
_SHAPE_PREDICTOR_PATH = str(BASE_DIR / 'shape_predictor_68_face_landmarks.dat')

# embadding模型全局加载加载
GPU_ID = int(settings.gpu.gpu_id)
option = fd.RuntimeOption()
option.use_gpu(GPU_ID)
embedding_model = fd.vision.faceid.ArcFace(str(BASE_DIR / 'ms1mv3_arcface_r100.onnx'),
                                           runtime_option=option)

# 定义全局变量，会被main.py初始化
GLOBAL_PROCESS_POOL: Optional[ProcessPoolExecutor] = None

# 定义子进程的全局变量
_mp_detector = None
_mp_predictor = None

def _init_dlib_worker():
    """
    子进程的初始化函数。
    当 ProcessPoolExecutor 创建一个新的子进程时，会立刻运行这个函数。
    在这里加载模型，确保每个进程有一份独立的模型，互不干扰。
    """
    global _mp_detector, _mp_predictor
    logger.info(f"[Worker PID: {os.getpid()}] 正在加载 Dlib 模型...")
    try:
        _mp_detector = dlib.get_frontal_face_detector()
        _mp_predictor = dlib.shape_predictor(_SHAPE_PREDICTOR_PATH)
        logger.info(f"[Worker PID: {os.getpid()}] Dlib 模型加载完成")
    except Exception as e:
        logger.info(f"[Worker PID: {os.getpid()}] 模型加载失败: {e}")


def _dlib_task_implementation(image: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[dict]]:
    """
    真实的dlib计算的函数的，运行在子进程中。
    """
    # 使用子进程局部的模型
    global _mp_detector, _mp_predictor
    tip = "人脸特征像素正常，可以使用" # 图像质量提示信息
    if _mp_detector is None or _mp_predictor is None:
        logger.error(f"[Worker PID: {os.getpid()}] Dlib 模型未加载，请检查是否子进程初始化失败")
        return None, None

    # 转灰度cpu计算
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # 1. 检测人脸cpu计算
    faces = _mp_detector(gray, 1)
    if not faces:
        return None, None

    # 选择最大人脸
    face = max(faces, key=lambda r: r.width() * r.height())
    (x, y, w, h) = (face.left(), face.top(), face.width(), face.height())

    if w < 200 or h < 200:
        tip = "人脸特征像素小于200x200，可能影响对比"

    logger.info(f"[dlib] 人脸像素大小: {w}x{h}")
    # 2. 关键点
    shape = _mp_predictor(gray, face)

    # _shape_to_np, _five_from_68, _align_by_5pts 应为纯函数
    lm68 = _shape_to_np(shape)
    lm5 = _five_from_68(lm68)
    aligned = _align_by_5pts(image, lm5, (112, 112))

    if not aligned.flags['C_CONTIGUOUS']:
        aligned = np.ascontiguousarray(aligned)

    bbox = {"x": int(x), "y": int(max(0, int(y - h * 0.4))), "w": int(w), "h": int(h)}

    return aligned, bbox, tip


async def detect_and_extract_face(image: np.ndarray):
    """
    主程序调用的入口。
    它会检查 GLOBAL_PROCESS_POOL 是否已被 main.py 初始化。
    """
    loop = asyncio.get_running_loop()

    if GLOBAL_PROCESS_POOL is None:
        # 如果池子没初始化（比如直接运行此脚本测试），降级为同步或报错
        raise RuntimeError("全局进程池未初始化，请检查main.py是否正确启动")

    try:
        # 提交给进程池
        return await loop.run_in_executor(
            GLOBAL_PROCESS_POOL,
            _dlib_task_implementation,
            image
        )
    except Exception as e:
        print(f"Process Pool Error: {e}")
        return None, None, None


# ArcFace 常用 5点模板（112x112）
_ARCFACE_5PTS = np.array([
    [38.2946, 51.6963],  # 左眼
    [73.5318, 51.5014],  # 右眼
    [56.0252, 71.7366],  # 鼻子
    [41.5493, 92.3655],  # 左嘴角
    [70.7299, 92.2041],  # 右嘴角
], dtype=np.float32)


def _shape_to_np(shape: dlib.full_object_detection) -> np.ndarray:
    return np.array([(shape.part(i).x, shape.part(i).y) for i in range(68)], dtype=np.float32)

def _five_from_68(landmarks68: np.ndarray) -> np.ndarray:
    # 人脸68点
    left_eye = landmarks68[36:42].mean(axis=0)
    right_eye = landmarks68[42:48].mean(axis=0)
    nose = landmarks68[30]
    left_mouth = landmarks68[48]
    right_mouth = landmarks68[54]
    return np.stack([left_eye, right_eye, nose, left_mouth, right_mouth]).astype(np.float32)

def _align_by_5pts(img: np.ndarray, pts: np.ndarray, size: Tuple[int,int]=(112,112)) -> np.ndarray:
    """
    5点对齐
    """
    dst = _ARCFACE_5PTS
    M = cv2.estimateAffinePartial2D(pts, dst, method=cv2.LMEDS)[0]
    aligned = cv2.warpAffine(img, M, size, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    return aligned

# 异步获取特征向量
async def get_embedding(face_aligned: np.ndarray) -> np.ndarray:
    """
    异步提取 512d 单位化向量（float32, L2norm==1）
    :param face_aligned: 对齐后的 112x112 人脸图像
    :return: 归一化后的 512维特征向量
    """
    # return await asyncio.to_thread(get_embedding_sync, face_aligned)
    emb =  await asyncio.to_thread(get_embedding_sync, face_aligned)
    # 归一化 方便点积计算
    emb_q = emb / (np.linalg.norm(emb) + 1e-12)
    return emb_q

def get_embedding_sync(face_aligned: np.ndarray) -> np.ndarray:
    """同步获取特征向量的方法"""
    result = embedding_model.predict(face_aligned)
    emb = np.asarray(result.embedding, dtype=np.float32)
    # 再做一次归一化
    n = np.linalg.norm(emb) + 1e-12
    emb = emb / n
    return emb

def find_best_match_embedding(emb_q: np.ndarray, candidate_docs: List[dict]) -> Tuple[float, Optional[dict]]:
    """
    在候选文档列表中寻找最大相似度
    返回: (最佳相似度, 最佳匹配文档)
    """
    db_vecs = []
    valid_docs = []

    for d in candidate_docs:
        e = d.get("embedding")
        if e is None: continue

        # BSON Binary -> bytes -> numpy
        if isinstance(e, Binary):
            e = bytes(e)
        vec = np.frombuffer(e, dtype=np.float32)

        if vec.size != 512: continue

        # 归一化
        n = np.linalg.norm(vec) + 1e-12
        vec = vec / n
        db_vecs.append(vec)
        valid_docs.append(d)

    if not db_vecs:
        return 0.0, None

    # 批量计算点积 (余弦相似度)
    # emb_q 假设已经归一化
    sims = np.dot(db_vecs, emb_q)
    best_idx = int(np.argmax(sims))
    best_sim = float(sims[best_idx])

    return best_sim, valid_docs[best_idx]



