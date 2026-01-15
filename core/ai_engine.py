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
import threading
from app.core.config import settings
from app.core.logger import get_logger
import warnings

# 过滤InsightFace库中NumPy的FutureWarning，这个警告其实不影响使用，但看着膈应
warnings.filterwarnings('ignore', category=FutureWarning, module='insightface')

logger = get_logger(__name__)


try:
    import insightface
    INSIGHTFACE_AVAILABLE = True
except (ImportError, Exception) as e:
    logger.debug(f"InsightFace不可用: {e}，将使用dlib（CPU计算）")
    INSIGHTFACE_AVAILABLE = False

BASE_DIR = Path(__file__).resolve().parent.parent
_SHAPE_PREDICTOR_PATH = str(BASE_DIR / 'ai_models' / 'shape_predictor_68_face_landmarks.dat')

# GPU模型延迟加载（避免子进程继承）
_embedding_model: Optional[fd.vision.faceid.ArcFace] = None
_embedding_model_lock = threading.Lock()
_embedding_model_loading = False

# 定义全局变量，会被main.py初始化
GLOBAL_PROCESS_POOL: Optional[ProcessPoolExecutor] = None

# 定义子进程的全局变量
_mp_detector = None  # InsightFace app或dlib检测器
_mp_predictor = None  # dlib关键点预测器（仅当使用dlib时）
_use_insightface = False  # 是否使用InsightFace

def _init_dlib_worker():
    """
    子进程的初始化函数。
    当 ProcessPoolExecutor 创建一个新的子进程时，会立刻运行这个函数。
    优先使用InsightFace（GPU加速），如果不可用则降级到dlib。
    """
    global _mp_detector, _mp_predictor, _use_insightface
    
    # 在子进程中再次检查InsightFace是否可用（因为子进程可能环境不同）
    insightface_available = False
    try:
        import insightface
        insightface_available = True
    except (ImportError, Exception):
        insightface_available = False
    
    if insightface_available:
        try:
            logger.info(f"[Worker PID: {os.getpid()}] 正在加载 InsightFace 模型（GPU加速）...")
            # InsightFace初始化（优先使用GPU，如果不可用则降级到CPU）
            # providers=['CUDAExecutionProvider', 'CPUExecutionProvider'] 优先GPU，降级CPU

            _mp_detector = insightface.app.FaceAnalysis(
                name='buffalo_l',  # 或 'buffalo_s' (更小更快)
                providers=['CUDAExecutionProvider', 'CPUExecutionProvider']  # 优先GPU，降级CPU
            )
            # ctx_id: -1表示CPU，0-N表示GPU编号
            # 先尝试GPU，如果失败会自动降级到CPU
            try:
                _mp_detector.prepare(ctx_id=0, det_size=(640, 640))  # ctx_id=0 表示GPU 0
                logger.info(f"[Worker PID: {os.getpid()}] InsightFace 模型加载完成（使用GPU）")
            except Exception as gpu_error:
                logger.warning(f"[Worker PID: {os.getpid()}] GPU不可用，降级到CPU: {gpu_error}")
                _mp_detector.prepare(ctx_id=-1, det_size=(640, 640))  # ctx_id=-1 表示CPU
                logger.info(f"[Worker PID: {os.getpid()}] InsightFace 模型加载完成（使用CPU）")
            _use_insightface = True
        except Exception as e:
            logger.warning(f"[Worker PID: {os.getpid()}] InsightFace 加载失败，降级到dlib: {e}", exc_info=True)
            _use_insightface = False
            _init_dlib_fallback()
    else:
        logger.info(f"[Worker PID: {os.getpid()}] InsightFace 不可用（缺少依赖或未安装），使用 dlib")
        _use_insightface = False
        _init_dlib_fallback()

def _init_dlib_fallback():
    """dlib降级方案（保持原有逻辑）"""
    global _mp_detector, _mp_predictor
    try:
        logger.info(f"[Worker PID: {os.getpid()}] 正在加载 Dlib 模型...")
        _mp_detector = dlib.get_frontal_face_detector()
        _mp_predictor = dlib.shape_predictor(_SHAPE_PREDICTOR_PATH)
        logger.info(f"[Worker PID: {os.getpid()}] Dlib 模型加载完成")
    except Exception as e:
        logger.error(f"[Worker PID: {os.getpid()}] Dlib 模型加载失败: {e}")
        raise


def _dlib_task_implementation(image: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[dict]]:
    """
    人脸检测和对齐实现（支持InsightFace和dlib）
    运行在子进程中。
    """
    import time
    t_start = time.time()

    global _mp_detector, _mp_predictor, _use_insightface
    tip = "人脸特征像素正常，可以使用"
    
    if _mp_detector is None:
        logger.error(f"[Worker PID: {os.getpid()}] 检测器未加载")
        return None, None, None

    if _use_insightface:
        # ========== InsightFace路径（GPU加速） ==========
        t1 = time.time()
        try:
            # InsightFace检测（一步完成检测+对齐）
            # 输入：BGR格式的numpy数组
            faces = _mp_detector.get(image)
    
            logger.info(f"[性能] InsightFace检测耗时: {(time.time()-t1)*1000:.2f}ms, 检测到{len(faces)}张人脸")

            if not faces:
                logger.info(f"[性能] 总耗时(无人脸): {(time.time()-t_start)*1000:.2f}ms")
                return None, None, None

            # 选择面积最大的人脸
            face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
            # logger.info(f"[InsightFace] 最大人脸坐标: {face}")

            # 提取边界框 [x1, y1, x2, y2]
            bbox_coords = face.bbox.astype(int)
            x, y, x2, y2 = bbox_coords[0], bbox_coords[1], bbox_coords[2], bbox_coords[3]
            w = x2 - x
            h = y2 - y

            if w < 200 or h < 200:
                tip = "人脸特征像素过低，或影响检测效果"

            logger.info(f"[InsightFace] 人脸像素大小: {w}x{h}")

            # InsightFace获取对齐后的人脸（112x112）
            # 使用face对象的get_norm_crop_img方法或者使用face_align工具
            try:
                # 方法1: 尝试使用face.get_norm_crop_img（如果存在）
                if hasattr(face, 'get_norm_crop_img') and callable(face.get_norm_crop_img):
                    aligned = face.get_norm_crop_img(image)
                # 方法2: 使用insightface.utils.face_align.norm_crop
                elif hasattr(insightface, 'utils') and hasattr(insightface.utils, 'face_align'):
                    from insightface.utils import face_align
                    aligned = face_align.norm_crop(image, landmark=face.kps)
                # 方法3: 使用cv2进行对齐（基于5个关键点）
                else:
                    # 从face.kps获取5个关键点（左眼、右眼、鼻尖、左嘴角、右嘴角）
                    landmark = face.kps.astype(np.float32)
                    # 标准的112x112对齐模板点（ArcFace对齐标准）
                    dst_points = np.array([
                        [38.2946, 51.6963],  # 左眼中心
                        [73.5318, 51.5014],  # 右眼中心
                        [56.0252, 71.7366],  # 鼻尖
                        [41.5493, 92.3655],  # 左嘴角
                        [70.7299, 92.2041]   # 右嘴角
                    ], dtype=np.float32)
                    # 使用相似变换（similarity transform）以保持比例
                    # 计算变换矩阵（基于前3个点：左眼、右眼、鼻尖）
                    transform_matrix = cv2.getAffineTransform(landmark[:3], dst_points[:3])
                    aligned = cv2.warpAffine(image, transform_matrix, (112, 112), borderValue=0)
            except Exception as align_error:
                logger.error(f"[Worker PID: {os.getpid()}] 人脸对齐失败: {align_error}", exc_info=True)
                return None, None, None

            if not aligned.flags['C_CONTIGUOUS']:
                aligned = np.ascontiguousarray(aligned)

            bbox = {"x": int(x), "y": int(max(0, int(y - h * 0.4))), "w": int(w), "h": int(h)}

            return aligned, bbox, tip

        except Exception as e:
            logger.error(f"[Worker PID: {os.getpid()}] InsightFace检测失败: {e}", exc_info=True)
            return None, None, None

    else:
        # ========== Dlib路径（原有逻辑，保持不变） ==========
        if _mp_predictor is None:
            logger.error(f"[Worker PID: {os.getpid()}] Dlib 模型未加载，请检查是否子进程初始化失败")
            return None, None, None

        # 转灰度cpu计算
        t0 = time.time()
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        logger.info(f"[性能] 灰度转换耗时: {(time.time()-t0)*1000:.2f}ms")

        # 1. 检测人脸cpu计算（upsample=0 避免无人脸时过多上采样导致响应缓慢）
        t1 = time.time()
        faces = _mp_detector(gray, 1)
        logger.info(f"[性能] 人脸检测耗时: {(time.time()-t1)*1000:.2f}ms, 检测到{len(faces)}张人脸")

        if not faces:
            logger.info(f"[性能] 总耗时(无人脸): {(time.time()-t_start)*1000:.2f}ms")
            return None, None, None

        # 选择最大人脸
        face = max(faces, key=lambda r: r.width() * r.height())
        (x, y, w, h) = (face.left(), face.top(), face.width(), face.height())

        if w < 200 or h < 200:
            tip = "人脸特征像素过低，或影响检测效果"

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

def _get_embedding_model() -> fd.vision.faceid.ArcFace:
    """
    延迟加载GPU模型（单例模式，线程安全）
    避免在模块导入时加载，防止ProcessPoolExecutor子进程继承GPU模型引用
    
    Returns:
        ArcFace模型实例
    """
    global _embedding_model, _embedding_model_loading
    
    # 双重检查锁定模式（Double-Checked Locking）
    if _embedding_model is not None:
        return _embedding_model
    
    with _embedding_model_lock:
        # 再次检查（避免多线程竞争）
        if _embedding_model is not None:
            return _embedding_model
        
        # 防止重复加载
        if _embedding_model_loading:
            # 如果正在加载，等待加载完成
            while _embedding_model_loading:
                _embedding_model_lock.release()
                import time
                time.sleep(0.01)  # 短暂等待
                _embedding_model_lock.acquire()
            if _embedding_model is not None:
                return _embedding_model
        
        try:
            _embedding_model_loading = True
            logger.info(f"[GPU模型] 进程 {os.getpid()} 开始延迟加载GPU模型...")
            
            GPU_ID = int(settings.gpu.gpu_id)
            option = fd.RuntimeOption()
            option.use_gpu(GPU_ID)
            _embedding_model = fd.vision.faceid.ArcFace(
                str(BASE_DIR / 'ai_models' / 'ms1mv3_arcface_r100.onnx'),
                runtime_option=option
            )
            
            logger.info(f"[GPU模型] 进程 {os.getpid()} GPU模型加载完成")
            return _embedding_model
            
        except Exception as e:
            logger.error(f"[GPU模型] 进程 {os.getpid()} GPU模型加载失败: {e}", exc_info=True)
            raise
        finally:
            _embedding_model_loading = False


def get_embedding_sync(face_aligned: np.ndarray) -> np.ndarray:
    """
    同步获取特征向量的方法
    使用延迟加载的GPU模型
    """
    embedding_model = _get_embedding_model()  # 延迟加载
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

        # 处理不同格式的embedding：
        # 1. numpy数组（从Redis缓存读取）
        if isinstance(e, np.ndarray):
            vec = e.astype(np.float32).flatten()
        # 2. BSON Binary类型（从MongoDB直接读取）
        elif isinstance(e, Binary):
            vec = np.frombuffer(bytes(e), dtype=np.float32)
        # 3. bytes类型
        elif isinstance(e, bytes):
            vec = np.frombuffer(e, dtype=np.float32)
        # 4. list类型（理论上不应该出现，但为了健壮性保留）
        elif isinstance(e, list):
            vec = np.array(e, dtype=np.float32)
        else:
            continue

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


def find_top_matches(emb_q: np.ndarray, candidate_docs: List[dict], top_k: int = 3, min_threshold: float = 0.0):
    """
    在候选文档列表中寻找相似度最高的 top_k 个匹配

    参数:
        emb_q: 查询 embedding（已归一化）
        candidate_docs: 候选文档列表
        top_k: 返回的最大数量
        min_threshold: 最小相似度阈值

    返回: List[(相似度, 文档)]，按相似度降序排列
    """
    db_vecs = []
    valid_docs = []

    for d in candidate_docs:
        e = d.get("embedding")
        if e is None:
            continue

        # 处理不同格式的embedding：
        # 1. numpy数组（从Redis缓存读取）
        if isinstance(e, np.ndarray):
            vec = e.astype(np.float32).flatten()
        # 2. BSON Binary类型（从MongoDB直接读取）
        elif isinstance(e, Binary):
            vec = np.frombuffer(bytes(e), dtype=np.float32)
        # 3. bytes类型
        elif isinstance(e, bytes):
            vec = np.frombuffer(e, dtype=np.float32)
        # 4. list类型（理论上不应该出现，但为了健壮性保留）
        elif isinstance(e, list):
            vec = np.array(e, dtype=np.float32)
        else:
            continue

        if vec.size != 512:
            continue

        # 归一化
        n = np.linalg.norm(vec) + 1e-12
        vec = vec / n
        db_vecs.append(vec)
        valid_docs.append(d)

    if not db_vecs:
        return []

    # 批量计算点积 (余弦相似度)
    sims = np.dot(db_vecs, emb_q)

    # 找到所有大于等于阈值的索引
    valid_indices = np.where(sims >= min_threshold)[0]

    if len(valid_indices) == 0:
        return []

    # 按相似度降序排序
    sorted_indices = valid_indices[np.argsort(-sims[valid_indices])]

    # 取前 top_k 个
    top_indices = sorted_indices[:top_k]

    # 返回 (相似度, 文档) 列表
    results = [(float(sims[idx]), valid_docs[idx]) for idx in top_indices]

    return results



