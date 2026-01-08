import uuid
import cv2
import numpy as np
from typing import List
from pathlib import Path
from bson.binary import Binary
from fastapi import APIRouter, Depends, HTTPException, Query, Body
from motor.motor_asyncio import AsyncIOMotorClient

from app.core.database import db
from app.core.database import get_session
from app.core.exceptions import DatabaseError
from app.core.config import settings
from app.services import person as person_crud
from app.utils.image_loader import base64_to_mat
from app.utils.utils_mongo import doc_to_person_read
from app.core.logger import get_logger
from app.core import ai_engine
from app.models.request.person_interface_req import *
from app.models.response.person_interface_rep import PersonFeatureResponse, PersonRead, PersonsFeatureResponse, SearchPersonResponse


logger = get_logger(__name__)

router = APIRouter(prefix="/persons", tags=["Persons Management"])
BASE_DIR = Path(__file__).resolve().parent.parent
MIN_FEATURE_IMAGE_HEIGHT_PX = int(settings.feature_image.min_feature_image_height_px)
MIN_FEATURE_IMAGE_WIDTH_PX = int(settings.feature_image.min_feature_image_width_px)

@router.post("",response_model=PersonFeatureResponse)
async def create_person_api(
        request: PersonFeatureRequest = Body(..., description="人脸特征录入请求体")
):
    """
    单个上传人物特征接口
    - 如果 number 已存在，则更新该人物信息
    - 如果 number 不存在，则创建新人物
    """
    try:
        # 1. 解析图片数据
        image_data, filename = await base64_to_mat(request.photo)

        if image_data is None or not isinstance(image_data, np.ndarray) or image_data.size == 0:
            logger.error(f"[/persons] 未接收到有效图片数据或图像数据存在异常")
            raise HTTPException(status_code=400, detail="未接收到有效图片数据或图像数据存在异常")

        # 2. 检测并裁剪人脸
        try:
            face_image, bbox , tip = await ai_engine.detect_and_extract_face(image_data)
        except Exception as e:
            logger.error(f"[/persons] 人脸检测服务内部错误: {e}")
            raise HTTPException(status_code=500, detail=f"人脸检测服务内部错误: {str(e)}")

        if face_image is None:
            logger.error(f"[/persons] 未检测到有效人脸")
            raise HTTPException(status_code=422, detail="未检测到有效人脸")

        if face_image.shape[0] < MIN_FEATURE_IMAGE_WIDTH_PX or face_image.shape[1] < MIN_FEATURE_IMAGE_HEIGHT_PX:
            logger.error(f"[/persons] 检测人脸特征尺寸过小，小于{MIN_FEATURE_IMAGE_WIDTH_PX}*{MIN_FEATURE_IMAGE_HEIGHT_PX}px")
            raise HTTPException(
                status_code=422,
                detail=f"检测到的人脸过小(小于{MIN_FEATURE_IMAGE_WIDTH_PX}*{MIN_FEATURE_IMAGE_HEIGHT_PX}px)，无法识别，请重新捕捉人脸"
            )

        # 3. 提取特征向量
        try:
            emb_q = await ai_engine.get_embedding(face_image)
        except Exception as e:
            logger.error(f"[/persons] 人脸特征提取失败: {e}")
            raise HTTPException(status_code=500, detail=f"人脸特征提取失败: {str(e)}")

        # 4. 保存裁剪后的人脸图片
        try:
            media_dir = BASE_DIR / "media" / "person_photos"
            media_dir.mkdir(parents=True, exist_ok=True)

            # 文件名
            filename = f"{request.name}_{request.number}_{uuid.uuid4().hex[:8]}.jpg"
            save_path = media_dir / filename
            cv2.imwrite(str(save_path), face_image)
        except Exception as e:
            logger.error(f"[/persons] 保存图片失败: {e}")
            raise HTTPException(status_code=500, detail=f"保存图片失败: {str(e)}")

        # 5. 保存到数据库
        bbox_str = f"{bbox['x']},{bbox['y']},{bbox['w']},{bbox['h']}" if bbox else ""

        person_dict = {
            "name": request.name,
            "number": request.number,
            "photo_path": f"/media/person_photos/{filename}",
            "bbox": bbox_str,  # 人脸检测框: x,y,w,h,
            "embedding": Binary(emb_q.tobytes()),
            "tip": tip if tip else ""
        }
        logger.info(f"person_dict prepared for DB: {person_dict}")

        try:
            # 使用 update_or_create_person 实现存在则更新，不存在则创建
            doc, is_updated = await person_crud.update_or_create_person(db, person_dict)
            action = "更新" if is_updated else "创建"
            logger.info(f"[/persons] 人物 {request.name} {action}成功")
        except Exception as e:
            logger.error(f"[/persons] 数据库操作失败: {e}")
            raise HTTPException(status_code=500, detail=f"数据库操作失败: {str(e)}")

        return PersonFeatureResponse(
            id=str(doc["_id"]),
            name=doc.get("name"),
            number=doc.get("number"),
            photo_path=doc.get("photo_path"),
            tip=doc.get("tip"),
        )

    except HTTPException:
        # 重新抛出 HTTPException
        raise
    except Exception as e:
        # 捕获所有未预期的异常
        logger.error(f"[/persons] 未预期的异常: {e}")
        raise HTTPException(status_code=500, detail=f"服务器内部错误: {str(e)}")

@router.post("/batch", response_model=PersonsFeatureResponse)
async def create_persons_batch_api(
        request: PersonsFeatureRequest = Body(..., description="批量人脸特征录入请求体")
):
    """
    批量上传人物特征接口
    接受多个人物的照片和信息，返回批量处理结果
    """
    results = []

    for idx, person_req in enumerate(request.persons):
        try:
            # 1. 解析图片数据
            image_data, filename = await base64_to_mat(person_req.photo)

            if image_data is None or not isinstance(image_data, np.ndarray) or image_data.size == 0:
                logger.error(f"[/persons/batch] 第{idx+1}个人物: 未接收到有效图片数据或图像数据存在异常")
                # 添加失败记录
                results.append(PersonFeatureResponse(
                    id="",
                    name=person_req.name,
                    number=person_req.number,
                    photo_path="",
                    tip=f"错误: 未接收到有效图片数据或图像数据存在异常"
                ))
                continue

            # 2. 检测并裁剪人脸
            try:
                face_image, bbox, tip = await ai_engine.detect_and_extract_face(image_data)
            except Exception as e:
                logger.error(f"[/persons/batch] 第{idx+1}个人物: 人脸检测服务内部错误: {e}")
                results.append(PersonFeatureResponse(
                    id="",
                    name=person_req.name,
                    number=person_req.number,
                    photo_path="",
                    tip=f"错误: 人脸检测服务内部错误 - {str(e)}"
                ))
                continue

            if face_image is None:
                logger.error(f"[/persons/batch] 第{idx+1}个人物: 未检测到有效人脸")
                results.append(PersonFeatureResponse(
                    id="",
                    name=person_req.name,
                    number=person_req.number,
                    photo_path="",
                    tip="错误: 未检测到有效人脸"
                ))
                continue

            if face_image.shape[0] < MIN_FEATURE_IMAGE_WIDTH_PX or face_image.shape[1] < MIN_FEATURE_IMAGE_HEIGHT_PX:
                logger.error(f"[/persons/batch] 第{idx+1}个人物: 检测人脸特征尺寸过小")
                results.append(PersonFeatureResponse(
                    id="",
                    name=person_req.name,
                    number=person_req.number,
                    photo_path="",
                    tip=f"错误: 检测到的人脸过小小于{MIN_FEATURE_IMAGE_WIDTH_PX}*{MIN_FEATURE_IMAGE_HEIGHT_PX}px"
                ))
                continue

            # 3. 提取特征向量
            try:
                emb_q = await ai_engine.get_embedding(face_image)
            except Exception as e:
                logger.error(f"[/persons/batch] 第{idx+1}个人物: 人脸特征提取失败: {e}")
                results.append(PersonFeatureResponse(
                    id="",
                    name=person_req.name,
                    number=person_req.number,
                    photo_path="",
                    tip=f"错误: 人脸特征提取失败 - {str(e)}"
                ))
                continue

            # 4. 保存裁剪后的人脸图片
            media_dir = BASE_DIR / "media" / "person_photos"
            media_dir.mkdir(parents=True, exist_ok=True)

            # 文件名
            filename = f"{person_req.name}_{person_req.number}_{uuid.uuid4().hex[:8]}.jpg"
            save_path = media_dir / filename
            cv2.imwrite(str(save_path), face_image)

            # 5. 保存到数据库
            # 将 bbox 字典转换为字符串格式 "x,y,w,h"
            bbox_str = f"{bbox['x']},{bbox['y']},{bbox['w']},{bbox['h']}" if bbox else ""

            person_dict = {
                "name": person_req.name,
                "number": person_req.number,
                "photo_path": f"/media/person_photos/{filename}",
                "bbox": bbox_str,  # 人脸检测框: x,y,w,h
                "embedding": Binary(emb_q.tobytes()),
                "tip": tip if tip else ""
            }

            try:
                # 使用 update_or_create_person 实现存在则更新，不存在则创建
                doc, is_updated = await person_crud.update_or_create_person(db, person_dict)
                # 添加成功记录
                action = "更新" if is_updated else "添加"
                results.append(PersonFeatureResponse(
                    id=str(doc["_id"]),
                    name=doc.get("name"),
                    number=doc.get("number"),
                    photo_path=doc.get("photo_path"),
                    tip=doc.get("tip", ""),
                ))
                logger.info(f"[/persons/batch] 第{idx+1}个人物: {person_req.name} {action}成功")
            except Exception as e:
                logger.error(f"[/persons/batch] 第{idx+1}个人物: 数据库操作失败: {e}")
                results.append(PersonFeatureResponse(
                    id="",
                    name=person_req.name,
                    number=person_req.number,
                    photo_path="",
                    tip=f"错误: 数据库操作失败 - {str(e)}"
                ))
                continue

        except Exception as e:
            # 捕获所有未预期的异常
            logger.error(f"[/persons/batch] 第{idx+1}个人物处理异常: {e}")
            results.append(PersonFeatureResponse(
                id="",
                name=person_req.name if hasattr(person_req, 'name') else "未知",
                number=person_req.number if hasattr(person_req, 'number') else "未知",
                photo_path="",
                tip=f"错误: 处理异常 - {str(e)}"
            ))

    return PersonsFeatureResponse(persons=results)

@router.get("", response_model=List[PersonRead])
async def read_persons_api(
        skip: int = Query(0, description="跳过数据条数"),
        limit: int = Query(100, description="返回数据条数")
):
    """
    获取人物列表接口
    - 支持分页查询
    """
    try:
        docs = await person_crud.get_persons(db, skip=skip, limit=limit)
        if not docs:
            logger.warning("[/persons(get)] 数据库为空，没有任何人物记录")
            raise HTTPException(status_code=404, detail="数据库为空，请先创建人物")
        return [doc_to_person_read(d) for d in docs]
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[/persons(get)] 查询失败: {e}")
        raise HTTPException(status_code=500, detail=f"查询人物列表失败: {str(e)}")


@router.post("/search", response_model=SearchPersonResponse)
async def search_person_api(
        request: SearchPersonRequest = Body(..., description="搜索人物请求体"),
) -> dict:
    """
    搜索人物接口
    - 只传 name: 模糊查询
    - 只传 number: 精确查询
    - 都传: 组合查询 (name模糊 AND number精确)
    """
    try:
        if not request.name and not request.number:
            raise HTTPException(status_code=400, detail="name 和 number 至少提供一个")

        persons_list = await person_crud.get_persons_list_dynamic(
            db, 
            name=request.name, 
            number=request.number,
            limit=20 
        )
        if not persons_list:
            logger.warning(f"[/persons/search] 未找到人物: {request.model_dump(exclude_unset=True)}")
            # raise HTTPException(status_code=404, detail="未找到符合条件的人物")
            return {"persons": []}

        # 4. 序列化列表：处理 ObjectId 并转为 Pydantic 模型
        result_data = []
        for p in persons_list:
            # 将MongoDB _id->id
            p_dict = {**p, "id": str(p["_id"])}
            # 使用 PersonRead 模型验证并转为 dict
            result_data.append(PersonRead(**p_dict).model_dump())
        return {
            "persons": result_data
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[/persons/search] 搜索失败: {e}")
        raise HTTPException(status_code=500, detail=f"搜索人物失败: {str(e)}")


@router.delete("/delete")
async def delete_person_general_api(
        request: DeletePersonRequest = Body(..., description="删除人物请求体"),
):
    """
    通用删除人物接口
    - 支持通过 name 或 id 删除
    """
    try:
        if not request.name and not request.id:
            raise HTTPException(status_code=400, detail="name 和 id 至少提供一个")

        if request.name:
            try:
                deleted_count = await person_crud.delete_persons_by_name(db, name_keyword=request.name)
                if deleted_count == 0:
                    logger.warning(f"[/persons/delete] 未找到匹配的人物: name={request.name}")
                    raise HTTPException(status_code=404, detail="未找到匹配人物")
                logger.info(f"[/persons/delete] 删除了 {deleted_count} 个人物: name={request.name}")
                return {"deleted_count": deleted_count, "message": f"成功删除 {deleted_count} 个人物"}
            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"[/persons/delete] 按姓名删除失败: {e}")
                raise HTTPException(status_code=500, detail=f"按姓名删除失败: {str(e)}")

        if request.id:
            try:
                deleted_person = await person_crud.delete_person_by_id(db, id=request.id)
                if not deleted_person:
                    logger.warning(f"[/persons/delete] 未找到该人物: id={request.id}")
                    raise HTTPException(status_code=404, detail="未找到该人物")
                logger.info(f"[/persons/delete] 删除人物成功: id={request.id}")
                return {"message": "人物已删除", "id": request.id}
            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"[/persons/delete] 按ID删除失败: {e}")
                raise HTTPException(status_code=500, detail=f"按ID删除失败: {str(e)}")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[/persons/delete] 删除操作失败: {e}")
        raise HTTPException(status_code=500, detail=f"删除操作失败: {str(e)}")


# 保留您原有的 delete 接口，以防前端有依赖
@router.delete("/by_name")
async def delete_persons_by_name_api(
        name: str = Query(..., min_length=1)
):
    """
    按姓名删除人物接口（模糊匹配）
    - 警告：此接口使用模糊匹配，可能误删同名人物
    - 例如：删除"张三"可能同时删除"张三丰"等包含该关键字的人物
    - 建议：使用 /delete 接口（支持精确匹配）或 /by_id 接口（ID唯一）
    - 保留接口，用于向后兼容
    """
    try:
        deleted_count = await person_crud.delete_persons_by_name(db, name_keyword=name)
        if deleted_count == 0:
            logger.warning(f"[/persons/by_name] 未找到匹配的人物: name={name}")
            raise HTTPException(status_code=404, detail="未找到匹配的人物")
        logger.info(f"[/persons/by_name] 删除了 {deleted_count} 个人物: name={name}")
        return {
            "deleted_count": deleted_count,
            "message": f"成功删除 {deleted_count} 个人物"
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[/persons/by_name] 删除失败: {e}")
        raise HTTPException(status_code=500, detail=f"删除操作失败: {str(e)}")


@router.delete("/by_id")
async def delete_person_by_id_api(
        request: DeletePersonByIdRequest = Body(..., description="按ID删除人物请求体")
):
    """
    按ID删除人物接口
    - 精确匹配人物ID
    - ID是唯一标识，不会出现误删
    """
    try:
        info = await person_crud.delete_person(db, person_id=request.id)
        if not info:
            logger.warning(f"[/persons/by_id] 人物不存在: id={request.id}")
            raise HTTPException(status_code=404, detail="人物不存在")
        logger.info(f"[/persons/by_id] 删除人物成功: id={request.id}")
        return {"message": "人物已删除", "id": info['_id']}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[/persons/by_id] 删除失败: {e}")
        raise HTTPException(status_code=500, detail=f"删除操作失败: {str(e)}")