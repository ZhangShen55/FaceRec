import uuid
import numpy as np
from pathlib import Path
from bson.binary import Binary
from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse
from motor.motor_asyncio import AsyncIOMotorDatabase

from app.core.database import get_session
from app.curd import person
from app.utils.image_loader import get_photo_mat
from app.core import ai_engine
from app.core.config import settings

# 定义 Router
router = APIRouter(tags=["Face Recognition"])

# 重新计算 BASE_DIR (为了保存图片)
BASE_DIR = Path(__file__).resolve().parent.parent.parent
FACETHRESH = settings.face.threshold


@router.post("/recognize")
async def recognize_face_api(
        photo=Depends(get_photo_mat),
        db: AsyncIOMotorDatabase = Depends(get_session),
):
    # 1) 拿图
    image_data, filename = (photo if isinstance(photo, tuple) else (photo, None))

    # 【防线 1：输入源头检查】
    if image_data is None:
        return JSONResponse(status_code=400, content={"message": "未接收到有效图片数据"})

    if not isinstance(image_data, np.ndarray) or image_data.size == 0:
        return JSONResponse(status_code=400, content={"message": "图片数据为空或格式错误"})

    # 【防线 2：内存连续性强制】
    if not image_data.flags['C_CONTIGUOUS']:
        image_data = np.ascontiguousarray(image_data)

    # 2) 检测+对齐
    try:
        # 注意：这里调用的是 ai_engine，它依赖 main.py 注入的进程池
        result = await ai_engine.detect_and_extract_face(image_data)
    except Exception as e:
        print(f"Face Detection Error: {e}")
        return JSONResponse(status_code=400, content={"message": "人脸检测服务内部错误"})

    if result is None:
        return JSONResponse(status_code=400, content={"message": "未检测到人脸"})

    face_image, bbox = result

    # 二次检查 result 解包后的内容
    if face_image is None:
        return JSONResponse(status_code=400, content={"message": "未检测到有效人脸"})

    # 3) 保存检测到的人脸裁剪图
    det_dir = BASE_DIR / "media" / "detections"
    det_dir.mkdir(parents=True, exist_ok=True)
    det_name = f"{uuid.uuid4().hex}.jpg"

    if isinstance(face_image, np.ndarray) and face_image.size > 0:
        # 实际生产中可能不需要每次都保存，视需求而定
        # cv2.imwrite(str(det_dir / det_name), face_image)
        pass

    detected_face_url = f"/media/detections/{det_name}"

    # 4) 取库里用于匹配的数据
    persons = await person.get_embeddings_for_match(db, limit=7000)
    if not persons:
        return {
            "has_face": True,
            "detected_face_url": detected_face_url,
            "bbox": bbox,
            "match": {
                "matched": False, "reason": "empty_gallery",
                "name": None, "similarity": None,
                "person_id": None, "photo_url": None,
            },
            "message": "人脸特征库为空，请先上传人脸数据",
            "filename": filename,
        }

    # 5) 当前图 embedding
    if not face_image.flags['C_CONTIGUOUS']:
        face_image = np.ascontiguousarray(face_image)

    if face_image.shape[0] < 10 or face_image.shape[1] < 10:
        return JSONResponse(status_code=400, content={"message": "检测到的人脸过小，无法识别"})

    try:
        emb_q = await ai_engine.get_embedding(face_image)
    except Exception as e:
        print(f"Embedding Error: {e}")
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