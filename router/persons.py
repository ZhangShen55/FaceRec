import uuid
import cv2
from typing import List
from pathlib import Path
from bson.binary import Binary
from fastapi import APIRouter, Depends, HTTPException, Form, Query
from motor.motor_asyncio import AsyncIOMotorClient
from sqlmodel import Session  # 如果您用到sqlmodel的话，虽然看起来主要是mongo

from app.core.database import get_session
from app.curd import person as person_crud
from app.schemas import schemas
from app.utils.image_loader import get_photo_mat
from app.core import ai_engine
from app.utils.utils_mongo import doc_to_person_read

router = APIRouter(prefix="/persons", tags=["Persons Management"])
BASE_DIR = Path(__file__).resolve().parent.parent.parent


@router.post("", response_model=schemas.PersonRead)
async def create_person_api(
        chinese_name: str = Form(...),
        description: str = Form(None),
        photo=Depends(get_photo_mat),
        db: AsyncIOMotorClient = Depends(get_session),
):
    if isinstance(photo, tuple):
        image, _ = photo
    else:
        image = photo

    # 2. 检测并裁剪人脸 (调用 ai_engine)
    result = await ai_engine.detect_and_extract_face(image)
    if result is None:
        raise HTTPException(status_code=1001, detail="未检测到人脸")

    face_image, _ = result
    if face_image is None:
        raise HTTPException(status_code=1001, detail="未检测到有效人脸")

    # 3. 提取特征向量
    embedding = await ai_engine.get_embedding(face_image)

    # 4. 保存裁剪后的人脸图片
    media_dir = BASE_DIR / "media" / "person_photos"
    media_dir.mkdir(parents=True, exist_ok=True)

    filename = f"{chinese_name}_{uuid.uuid4().hex}.jpg"
    save_path = media_dir / filename
    cv2.imwrite(str(save_path), face_image)

    person_dict = {
        "chinese_name": chinese_name,
        "description": description,
        "photo_path": f"/media/person_photos/{filename}",
        "embedding": Binary(embedding.tobytes()),
    }

    doc = await person_crud.create_person(db, person_dict)
    return doc_to_person_read(doc)


@router.get("", response_model=List[schemas.PersonRead])
async def read_persons_api(skip: int = 0, limit: int = 100, db=Depends(get_session)):
    docs = await person_crud.get_persons(db, skip=skip, limit=limit)
    return [doc_to_person_read(d) for d in docs]


@router.get("/search")
async def search_person_api(
        name: str = None,
        person_id: str = None,
        db: AsyncIOMotorClient = Depends(get_session)
):
    if name:
        persons = person_crud.get_persons_by_name(db, name_keyword=name)
        if not persons:
            raise HTTPException(status_code=404, detail="未找到匹配人物")
        return {"persons": [{"id": p.id, "name": p.chinese_name, "photo_url": p.photo_path} for p in persons]}

    if person_id:
        person = person_crud.get_person_by_id(db, person_id=person_id)
        if not person:
            raise HTTPException(status_code=404, detail="未找到该人物")
        return {"persons": [{"id": person.id, "name": person.chinese_name, "photo_url": person.photo_path}]}

    raise HTTPException(status_code=400, detail="请输入姓名或ID进行查询")


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
                "deleted": [{"id": p.id, "name": p.chinese_name} for p in deleted_persons]}

    if person_id:
        deleted_person = person_crud.delete_person_by_id(db, person_id=person_id)
        if not deleted_person:
            raise HTTPException(status_code=404, detail="未找到该人物")
        return {"message": f"人物 {deleted_person.chinese_name} 已删除"}

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
        "deleted": [{"id": p.id, "chinese_name": p.chinese_name} for p in deleted]
    }


@router.delete("/{person_id}")
async def delete_person_api(person_id: str, db=Depends(get_session)):
    info = await person_crud.delete_person(db, person_id=person_id)
    if not info:
        raise HTTPException(status_code=404, detail="人物不存在")
    return {"message": f"人物 {info['id']} 已删除"}