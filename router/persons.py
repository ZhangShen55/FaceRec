import uuid
import cv2
from typing import List, Optional
from pathlib import Path
from bson.binary import Binary
from fastapi import APIRouter, Depends, HTTPException, Form, Query
from fastapi.responses import JSONResponse
from motor.motor_asyncio import AsyncIOMotorClient

from app.core.database import get_session
from app.core.exceptions import DatabaseError
from app.curd import person as person_crud
from app.schemas import schemas
from app.utils.image_loader import get_photo_mat
from app.core import ai_engine
from app.utils.utils_mongo import doc_to_person_read
from app.core.logger import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/persons", tags=["Persons Management"])
BASE_DIR = Path(__file__).resolve().parent.parent

@router.post("")
async def create_person_api(
        name: str = Form(...),
        number: str = Form(...),
        photo=Depends(get_photo_mat),
        db: AsyncIOMotorClient = Depends(get_session),
):
    if isinstance(photo, tuple):
        image, _ = photo
    else:
        image = photo

    # 2. 检测并裁剪人脸
    try:
        # 注意：这里调用的是ai_engine，它依赖 main.py 注入的进程池
        result = await ai_engine.detect_and_extract_face(image)
    except Exception as e:
        logger.error(f"人脸检测服务内部错误: {e}")
        return JSONResponse(status_code=200, content={"message": "人脸检测服务内部错误"})
    if result is None:
        return JSONResponse(status_code=200, content={"code": 1001, "message": "未能够检测到人脸"})
    if result is None:
        return JSONResponse(status_code=200, content={"message": "画面中未能检测到人脸，请重新捕捉人脸"})

    face_image, _ , tip = result
    if face_image is None:
        return JSONResponse(status_code=200, content={"code": 1002, "message": "无效人脸特征"})

    # 3. 提取特征向量
    embedding = await ai_engine.get_embedding(face_image)

    # 4. 保存裁剪后的人脸图片
    media_dir = BASE_DIR / "media" / "person_photos"
    media_dir.mkdir(parents=True, exist_ok=True)

    # 文件名
    filename = f"{name}_{number}_{uuid.uuid4().hex[:8]}.jpg"
    save_path = media_dir / filename
    cv2.imwrite(str(save_path), face_image)

    person_dict = {
        "name": name,
        "number": number,
        "photo_path": f"/media/person_photos/{filename}",
        "embedding": Binary(embedding.tobytes())
    }
    try:
        doc = await person_crud.create_person(db, person_dict)
    except DatabaseError as e:
        logger.error(f"数据库操作失败: {e}")
        return JSONResponse(status_code=200, content={"code":400,"message": f"{e.detail}"})

    return {
        "id" : str(doc["_id"]),
        "name": doc.get("name"),
        "number": doc.get("number"),
        "photo_path": doc.get("photo_path"),
        "tip": tip
    }


@router.get("", response_model=List[schemas.PersonRead])
async def read_persons_api(skip: int = 0, limit: int = 100, db=Depends(get_session)):
    docs = await person_crud.get_persons(db, skip=skip, limit=limit)

    if not docs:
        logger.error("数据库为空，请先创建人物")
        return JSONResponse(status_code=200, content={"code":400,"message": f"数据库为空，请先创建人物"})

    return [doc_to_person_read(d) for d in docs]


@router.get("/search", response_model=dict)
async def search_person_api(
    name: str = Query(..., description="姓名，模糊匹配"),
    number: str = Query(..., description="编号，精确匹配"),
    db: AsyncIOMotorClient = Depends(get_session),
) -> dict:
    """
    强制使用 name(模糊) + number(精确) 组合查询，结果唯一。
    """
    person = await person_crud.get_persons_by_name_and_number(db, name=name, number=number)
    if not person:
        raise HTTPException(404, "未找到符合条件的人物")

    return {
        "person": schemas.PersonRead(**{**person, "id": str(person["_id"])}).model_dump()
    }


@router.delete("/delete")
async def delete_person_general_api(
        name: str = None,
        person_id: str = None,
        db: AsyncIOMotorClient = Depends(get_session)
):
    if name:
        deleted_persons = person_crud.delete_persons_by_name(db, name_keyword=name)
        if not deleted_persons:
            raise HTTPException(status_code=404, detail="未找到匹配人物")
        return {"deleted_count": len(deleted_persons),
                "deleted": [{"id": p.id, "name": p.name} for p in deleted_persons]}

    if person_id:
        deleted_person = person_crud.delete_person_by_id(db, person_id=person_id)
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