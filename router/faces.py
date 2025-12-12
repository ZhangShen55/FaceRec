import uuid
import numpy as np
from pathlib import Path
from fastapi import APIRouter, Body, HTTPException
from app.services import person
from app.utils.image_loader import base64_to_mat
from app.core import ai_engine
from app.core.config import settings
from app.core.database import db
from app.models.request.face_interface_req import PersonRecognizeRequest
from app.models.response.face_interface_rep import RecognizeResp
from app.core.logger import get_logger

logger = get_logger(__name__)

router = APIRouter(tags=["Face Recognition"])

BASE_DIR = Path(__file__).resolve().parent.parent
THRESHOLD = settings.face.threshold
CANDIDATE_THRESHOLD = settings.face.candidate_threshold
REC_MIN_FACE_HW = int(settings.face.rec_min_face_hw)

REASON_MAP = {
    "priority": "优先比对命中",
    "global": "全局比对命中",
}

@router.post("/recognize",response_model=RecognizeResp)
async def recognize_face_api(request: PersonRecognizeRequest = Body(..., description="人脸识别请求")):
    logger.debug(f"[recognize] 接收到请求参数：{request}")
    # 1. 解析图片数据
    image_data, filename = await base64_to_mat(request.photo)

    if image_data is None or not isinstance(image_data, np.ndarray) or image_data.size == 0:
        logger.error(f"[recognize] 未接收到有效图片数据或图像数据存在异常")
        raise HTTPException(status_code=400, detail="[recognize] 未接收到有效图片数据或图像数据存在异常")

    try:
        face_image, bbox , _ = await ai_engine.detect_and_extract_face(image_data)
    except Exception as e:
        logger.error(f"[recognize] 人脸检测服务内部错误: {e}")
        raise HTTPException(status_code=401, detail="[recognize] 人脸检测服务内部错误")

    if face_image is None:
        logger.error(f"[recognize] 未检测到有效人脸")
        raise HTTPException(status_code=402, detail="[recognize] 未检测到有效人脸")

    if face_image.shape[0] < REC_MIN_FACE_HW or face_image.shape[1] < REC_MIN_FACE_HW:
        # 默认最小10*10
        logger.error(f"[recognize] 检测到的人脸过小小于{REC_MIN_FACE_HW}*{REC_MIN_FACE_HW}px，无法识别，请重新捕捉人脸")
        raise HTTPException(status_code=403, detail=f"[recognize] 检测到的人脸过小小于{REC_MIN_FACE_HW}*{REC_MIN_FACE_HW}*px，无法识别，请重新捕捉人脸")

    try:
        # 获取到检测图像的embedding
        emb_q = await ai_engine.get_embedding(face_image)
        # 归一化 为点积计算
        # emb_q = emb_q / (np.linalg.norm(emb_q) + 1e-12)
    except Exception as e:
        logger.error(f"[recognize] 人脸特征提取失败: {e}")
        raise HTTPException(status_code=404, detail="[recognize] 人脸特征提取失败")

    det_name = f"{uuid.uuid4().hex}.jpg"
    detected_face_url = f"/media/detections/{det_name}"

    # 优先与候选的persons人物进行对比
    CANDIDATE_THRESHOLD = 0.2
    targets = request.targets
    if targets:
        logger.info("[recognize] 优先targets比对...")
        # 1. 获取指定候选人的特征
        candidate_docs = await person.get_targets_embeddings(db, targets)

        logger.info(f"[recognize] 传递排课人数:{len(targets)} -> 成功匹配到的人数: {len(candidate_docs)}")

        if candidate_docs:
            # 2. 比对
            best_sim, best_doc = ai_engine.find_best_match_embedding(emb_q, candidate_docs)

            # 3. 判断结果
            if best_sim > CANDIDATE_THRESHOLD:
                logger.info(f"[recognize] 优先比对命中: {best_doc.get('name')} (Sim: {best_sim})")
                return RecognizeResp.from_recognize(
                    matched=True,has_face=True,
                    bbox=bbox,
                    best_sim=best_sim,
                    threshold=CANDIDATE_THRESHOLD,
                    reason=REASON_MAP["priority"],
                    person_doc=best_doc
                )

        logger.info("[recognize] 优先persons比对未命中，转入全局比对...")
    # 全局比对
    logger.info("[recognize] 全局比对...")
    all_docs = await person.get_embeddings_for_match(db)

    if not all_docs:
        logger.info("[recognize] (全局)数据库中没有有效人脸特征")
        raise HTTPException(status_code=405, detail="[recognize] (全局)数据库中没有有效人脸特征,请先录入人脸数据")

    # 2. 比对
    best_sim, best_doc = ai_engine.find_best_match_embedding(emb_q, all_docs)
    logger.info(f"[recognize] 全局匹配best结果: {best_doc.get('name')},{best_doc.get('number')},(Sim: {best_sim})")
    threshold = request.threshold if request.threshold else THRESHOLD

    # 3. 判断结果 (使用严格阈值)
    if best_sim >= threshold:
        logger.info(f"[recognize] 全局比对命中: {best_doc.get('name')} (Sim: {best_sim})")
        return RecognizeResp.from_recognize(
            matched=True,
            has_face=True if bbox else False,
            bbox=bbox,
            best_sim=best_sim,
            threshold=threshold,
            reason=REASON_MAP["global"],
            person_doc=best_doc
        )
    else:
        logger.info(f"[recognize] 全局比对失败，最高分: {best_sim} 低于阈值: {threshold}")
        return RecognizeResp.from_recognize(
            matched=False,
            has_face=True if bbox else False,
            bbox=bbox,
            best_sim=best_sim,
            threshold=threshold,
            reason="相似度低于阈值，与已知人脸库不匹配",
            person_doc=best_doc
        )