import uuid
import cv2
import numpy as np
from typing import List
from pathlib import Path
from bson.binary import Binary
from fastapi import APIRouter, Depends, HTTPException, Form, Query, Body
from fastapi.responses import JSONResponse
from motor.motor_asyncio import AsyncIOMotorClient

from app.core.database import db
from app.core.database import get_session
from app.core.exceptions import DatabaseError
from app.core.config import settings
from app.services import person as person_crud
from app.models import schemas
from app.utils.image_loader import get_photo_mat, base64_to_mat
from app.utils.utils_mongo import doc_to_person_read
from app.core.logger import get_logger
from app.core import ai_engine
from app.models.request.person_interface_req import *
from app.models.response.person_interface_rep import PersonFeatureResponse, PersonRead



logger = get_logger(__name__)

router = APIRouter(prefix="/persons", tags=["Persons Management"])
BASE_DIR = Path(__file__).resolve().parent.parent
MIN_FEATURE_IMAGE_HEIGHT_PX = int(settings.feature_image.min_feature_image_height_px)
MIN_FEATURE_IMAGE_WIDTH_PX = int(settings.feature_image.min_feature_image_width_px)

@router.post("",response_model=PersonFeatureResponse)
async def create_person_api(
        request: PersonFeatureRequest = Body(...,default="人脸特征录入请求体")
):
    # 1. 解析图片数据
    image_data, filename = await base64_to_mat(request.photo)

    if image_data is None or not isinstance(image_data, np.ndarray) or image_data.size == 0:
        logger.error(f"[/persons] 未接收到有效图片数据或图像数据存在异常")
        raise HTTPException(status_code=400, detail="[/persons] 未接收到有效图片数据或图像数据存在异常")
    # 2. 检测并裁剪人脸
    try:
        face_image, bbox , tip = await ai_engine.detect_and_extract_face(image_data)
    except Exception as e:
        logger.error(f"[/persons] 人脸检测服务内部错误: {e}")
        raise HTTPException(status_code=401, detail="[/persons] 人脸检测服务内部错误")
    if face_image is None:
        logger.error(f"[/persons] 未检测到有效人脸")
        raise HTTPException(status_code=402, detail="[/persons] 未检测到有效人脸")

    if face_image.shape[0] < MIN_FEATURE_IMAGE_WIDTH_PX or face_image.shape[1] < MIN_FEATURE_IMAGE_HEIGHT_PX:
        # 默认最小10*10
        logger.error(f"[/persons] 检测人脸特征尺寸过小，小于{MIN_FEATURE_IMAGE_WIDTH_PX}*{MIN_FEATURE_IMAGE_HEIGHT_PX}px，无法提取特征，请重新上传人脸图片")
        raise HTTPException(status_code=403, detail=f"[/persons] 检测到的人脸过小小于{MIN_FEATURE_IMAGE_WIDTH_PX}*{MIN_FEATURE_IMAGE_HEIGHT_PX}*px，无法识别，请重新捕捉人脸")

    # 3. 提取特征向量
    try:
        emb_q = await ai_engine.get_embedding(face_image)
        # 归一化 已经在ai_engine.get_embedding中实现
    except Exception as e:
        logger.error(f"[/persons] 人脸特征提取失败: {e}")
        raise HTTPException(status_code=404, detail="[/persons] 人脸特征提取失败")

    # 4. 保存裁剪后的人脸图片
    media_dir = BASE_DIR / "media" / "person_photos"
    media_dir.mkdir(parents=True, exist_ok=True)

    # 文件名
    filename = f"{request.name}_{request.number}_{uuid.uuid4().hex[:8]}.jpg"
    save_path = media_dir / filename
    cv2.imwrite(str(save_path), face_image)

    person_dict = {
        "name": request.name,
        "number": request.number,
        "photo_path": f"/media/person_photos/{filename}",
        "embedding": Binary(emb_q.tobytes()),
        "tip": tip if tip else ""
    }
    try:
        doc = await person_crud.create_person(db, person_dict)
    except DatabaseError as e:
        logger.error(f"[/persons] 数据库操作失败,人物重复添加: {e}")
        raise HTTPException(status_code=405, detail=f"[/persons] 数据库操作失败，人物重复添加: {e}")

    return PersonFeatureResponse(
        id=str(doc["_id"]),
        name=doc.get("name"),
        number=doc.get("number"),
        photo_path=doc.get("photo_path"),
        tip=doc.get("tip"),
    )

@router.get("", response_model=List[PersonRead])
async def read_persons_api(request: GetPersonListRequest = Body(...,default="获取人物列表请求体")):
    docs = await person_crud.get_persons(db, skip=request.skip, limit=request.limit)

    if not docs:
        logger.error("[/persons(get)] 数据库为空，请先创建人物")
        raise HTTPException(status_code=404, detail="[/persons(get)] 数据库为空，请先创建人物")

    return [doc_to_person_read(d) for d in docs]


@router.get("/search", response_model=dict)
async def search_person_api(
        request: SearchPersonRequest = Body(...,default="搜索人物请求体"),
    name: str = Query(..., description="姓名，模糊匹配"),
    number: str = Query(..., description="编号，精确匹配")
) -> dict:
    """
    强制使用 name(模糊) + number(精确) 组合查询，结果唯一。
    """
    person = await person_crud.get_persons_by_name_and_number(db, name=request.name, number=request.number)
    if not person:
        raise HTTPException(404, "未找到符合条件的人物")

    return {
        "person": PersonRead(**{**person, "id": str(person["_id"])}).model_dump()
    }


@router.delete("/delete")
async def delete_person_general_api(
        request: DeletePersonRequest = Body(...,default="删除人物请求体"),
        # name: str = None,
        # person_id: str = None,
):
    if request.name:
        deleted_persons = person_crud.delete_persons_by_name(db, name_keyword=name)
        if not deleted_persons:
            raise HTTPException(status_code=404, detail="未找到匹配人物")
        return {"deleted_count": len(deleted_persons),
                "deleted": [{"id": p.id, "name": p.name} for p in deleted_persons]}

    if request.id:
        deleted_person = person_crud.delete_person_by_id(db, id=request.id)
        if not deleted_person:
            raise HTTPException(status_code=404, detail="未找到该人物")
        return {"message": f"人物 {deleted_person.name} 已删除"}

    raise HTTPException(status_code=400, detail="请输入姓名或ID进行删除")


# 保留您原有的 delete 接口，以防前端有依赖
@router.delete("/by_name")
async def delete_persons_by_name_api(
        name: str = Query(..., min_length=1),
        db: AsyncIOMotorClient = Depends(get_session)
):
    deleted = person_crud.delete_persons_by_name(db, name_keyword=name)
    if not deleted:
        raise HTTPException(status_code=404, detail="未找到匹配的人物")
    return {
        "deleted_count": len(deleted),
        "deleted": [{"id": p.id, "name": p.name} for p in deleted]
    }


@router.delete("/{person_id}")
async def delete_person_api(person_id: str, db=Depends(get_session)):
    info = await person_crud.delete_person(db, person_id=person_id)
    if not info:
        raise HTTPException(status_code=404, detail="人物不存在")
    return {"message": f"人物 {info['id']} 已删除"}