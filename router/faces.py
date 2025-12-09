import uuid
import numpy as np
from pathlib import Path
from typing import List, Optional
from pydantic import Json
from bson.binary import Binary
from fastapi import APIRouter, Depends, Body, HTTPException, Form
from fastapi.responses import JSONResponse
from motor.motor_asyncio import AsyncIOMotorDatabase

from app.core.database import get_session
from app.curd import person
from app.utils.image_loader import get_photo_mat
from app.core import ai_engine
from app.core.config import settings
from app.schemas.schemas import PersonBase
from app.core.logger import get_logger

logger = get_logger(__name__)

router = APIRouter(tags=["Face Recognition"])

BASE_DIR = Path(__file__).resolve().parent.parent
FACETHRESH = settings.face.threshold

@router.post("/recognize")
async def recognize_face_api(
        photo=Depends(get_photo_mat),
        persons: List[PersonBase] = Body(default=[], description="可能被排课老师对象"),
        db: AsyncIOMotorDatabase = Depends(get_session),
):
    logger.debug(f"接受到请求：{photo}")
    image_data, filename = (photo if isinstance(photo, tuple) else (photo, None))

    if image_data is None:
        return JSONResponse(status_code=200, content={"message": "未接收到有效图片数据"})

    if not isinstance(image_data, np.ndarray) or image_data.size == 0:
        return JSONResponse(status_code=200, content={"message": "图片数据为空或格式错误"})

    # 内存连续性强制要求
    if not image_data.flags['C_CONTIGUOUS']:
        image_data = np.ascontiguousarray(image_data)

    # 检测&对齐人脸
    try:
        # 注意：这里调用的是ai_engine，它依赖 main.py 注入的进程池
        result = await ai_engine.detect_and_extract_face(image_data)
        # TODO yolo训练一个人脸姿态的问题，用于给业务端反馈人脸姿态问题（暂时不做）
    except Exception as e:
        logger.error(f"人脸检测服务内部错误: {e}")
        return JSONResponse(status_code=200, content={"message": "人脸检测服务内部错误"})

    if result is None:
        return JSONResponse(status_code=200, content={"message": "画面中未能检测到人脸，请重新捕捉人脸！"})

    face_image, bbox , _ = result

    # 再次检查result解包后的内容
    if face_image is None:
        return JSONResponse(status_code=200, content={"message": "未检测到有效人脸"})

    # 3) 保存检测到的人脸裁剪图，主要是用于前端显示
    det_dir = BASE_DIR / "media" / "detections"
    det_dir.mkdir(parents=True, exist_ok=True)
    det_name = f"{uuid.uuid4().hex}.jpg"

    if isinstance(face_image, np.ndarray) and face_image.size > 0:
        # 测试阶段不需要保存图片，压测炸存储
        # cv2.imwrite(str(det_dir / det_name), face_image)
        pass
    detected_face_url = f"/media/detections/{det_name}"

    # 4) 取库里用于匹配的数据
    persons = await person.get_embeddings_for_match(db, limit=10000)
    if not persons:
        return {
            "has_face": True,
            "detected_face_url": detected_face_url,
            "bbox": bbox,
            "match": {
                "matched": False,
                "reason": "人脸特征库为空",
                "person_id": None,
                "name": None,
                "number": None,
                "similarity": None,
                # "photo_url": None, # 业务不需要
            },
            "message": "人脸特征库为空，请先上传人脸数据",
            "filename": filename,
        }

    # 5) 当前图 embedding
    if not face_image.flags['C_CONTIGUOUS']:
        face_image = np.ascontiguousarray(face_image)

    if face_image.shape[0] < 10 or face_image.shape[1] < 10:
        return JSONResponse(status_code=200, content={"message": "检测到的人脸过小，无法识别"})

    try:
        emb_q = await ai_engine.get_embedding(face_image)
    except Exception as e:
        logger.error(f"人脸特征提取失败: {e}")
        return JSONResponse(status_code=500, content={"message": "特征提取失败"})

    # 6) 逐库比对
    db_vecs, metas = [], []
    for d in persons:
        e = d.get("embedding")
        if e is None: continue
        if isinstance(e, Binary):
            e = bytes(e)
        vec = np.frombuffer(e, dtype=np.float32)
        if vec.size != 512: continue

        n = np.linalg.norm(vec) + 1e-12
        vec = vec / n
        db_vecs.append(vec)
        metas.append(d)

    if not db_vecs:
        return {
            "has_face": True,
            "detected_face_url": detected_face_url,
            "bbox": bbox,
            "match": {
                "matched": False, "reason": "empty_gallery",
                "name": None, "similarity": None,
                "person_id": None, "photo_url": None,
            },
            "message": "人物库没有有效人脸特征",
            "filename": filename,
        }

    sims = [float(np.dot(emb_q, v)) for v in db_vecs]
    best_idx = int(np.argmax(sims))
    best_sim = sims[best_idx]

    if best_sim < FACETHRESH:
        match_payload = {
            "matched": False, "reason": "no_match",
            "name": None, "similarity": f"{best_sim * 100:.2f}%",
            "person_id": None, "photo_url": None,
        }
        message = "未匹配到该人物"
    else:
        p = metas[best_idx]
        match_payload = {
            "matched": True, "reason": "ok",
            "name": p.get("chinese_name"),
            "similarity": f"{best_sim * 100:.2f}%",
            "person_id": str(p["_id"]),
            "photo_url": p.get("photo_path"),
        }
        message = "识别成功"

    return {
        "has_face": True,
        "detected_face_url": detected_face_url,
        "bbox": bbox,
        "match": match_payload,
        "message": message,
        "filename": filename,
    }


@router.post("/recognize2")
async def recognize_face_api2(
        photo=Depends(get_photo_mat),
        db: AsyncIOMotorDatabase = Depends(get_session),
        persons: List[PersonBase] = Body(default=[], description="可能被排课老师对象")
):
    # logger.info(f"person: {persons}")
    # logger.info(f"接受到请求，候选人数: {len(persons)}")
    image_data, filename = (photo if isinstance(photo, tuple) else (photo, None))

    if image_data is None:
        raise HTTPException(status_code=200, content={"message": "未接收到有效图片数据"})

    if not isinstance(image_data, np.ndarray) or image_data.size == 0:
        raise HTTPException(status_code=200, content={"message": "图片数据为空或格式错误"})

    if not image_data.flags['C_CONTIGUOUS']:
        image_data = np.ascontiguousarray(image_data)

    try:
        result = await ai_engine.detect_and_extract_face(image_data)
    except Exception as e:
        logger.error(f"人脸检测服务内部错误: {e}")
        raise HTTPException(status_code=200, content={"message": "人脸检测服务内部错误"})

    if result is None:
        raise HTTPException(status_code=200, content={"message": "画面中未能检测到人脸，请重新捕捉人脸！"})

    face_image, bbox , _ = result

    if face_image is None:
        raise HTTPException(status_code=200, content={"message": "未检测到有效人脸"})
    # --- C. 提取当前人脸特征 (提前到这里) ---
    # 必须先拿到特征，才能去和数据库比对
    if not face_image.flags['C_CONTIGUOUS']:
        face_image = np.ascontiguousarray(face_image)

    if face_image.shape[0] < 10 or face_image.shape[1] < 10:
        raise HTTPException(status_code=200, content={"message": "检测到的人脸过小，无法识别"})

    try:
        emb_q = await ai_engine.get_embedding(face_image)
        # 确保 emb_q 归一化，方便后续点积计算
        emb_q = emb_q / (np.linalg.norm(emb_q) + 1e-12)
    except Exception as e:
        logger.error(f"人脸特征提取失败: {e}")
        raise HTTPException(status_code=500, content={"message": "特征提取失败"})

    # --- D. 准备返回结构的基础数据 ---
    # 保存检测图片 (仅生成路径，不实际写入以节省IO，除非调试)
    det_name = f"{uuid.uuid4().hex}.jpg"
    detected_face_url = f"/media/detections/{det_name}"

    # 定义统一的成功返回闭包
    def create_response(matched: bool, reason: str, person_doc: dict = None, score: float = 0.0):
        match_data = {
            "matched": matched,
            "reason": reason,
            "person_id": str(person_doc["_id"]) if person_doc else None,
            "name": person_doc.get("name") if person_doc else None,  # 改为 name
            "number": person_doc.get("number") if person_doc else None,  # 改为 number
            "similarity": f"{score * 100:.2f}%" if score else None,
        }
        return {
            "has_face": True,
            # "detected_face_url": detected_face_url,
            "bbox": bbox,
            "match": match_data,
            "message": "识别成功" if matched else "未匹配到该人物",
            "filename": filename,
        }

    # --- E. 策略1：优先比对 (Priority Match) ---
    CANDIDATE_THRESHOLD = 0.2

    if persons:
        # 1. 获取指定候选人的特征
        candidate_docs = await person.get_persons_embeddings(db, persons)

        logger.info(f"候选人数: {len(candidate_docs)}")

        if candidate_docs:
            # 2. 比对
            best_sim, best_doc = ai_engine.find_best_match_embedding(emb_q, candidate_docs)

            # 3. 判断结果
            if best_sim > CANDIDATE_THRESHOLD:
                logger.info(f"优先比对命中: {best_doc.get('name')} (Sim: {best_sim})")
                return create_response(True, "priority_match", best_doc, best_sim)

        logger.info("优先比对未命中，转入全局比对...")

    # --- F. 策略2：全局比对 (Global Match) ---
    # 1. 获取全量特征 (limit 限制防止内存溢出)
    all_docs = await person.get_embeddings_for_match(db, limit=10000)

    if not all_docs:
        return create_response(False, "empty_gallery")

    # 2. 比对
    best_sim, best_doc = ai_engine.find_best_match_embedding(emb_q, all_docs)

    # 3. 判断结果 (使用严格阈值)
    if best_sim >= FACETHRESH:
        logger.info(f"全局比对命中: {best_doc.get('name')} (Sim: {best_sim})")
        return create_response(True, "ok", best_doc, best_sim)
    else:
        logger.info(f"全局比对失败，最高分: {best_sim}")
        return create_response(False, "no_match", None, best_sim)