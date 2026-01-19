import uuid
import numpy as np
from pathlib import Path
from fastapi import APIRouter, Body, HTTPException
from app.services import person
from app.services.cache_service import cache_service
from app.utils.image_loader import base64_to_mat
from app.core import ai_engine
from app.core.config import settings
from app.core.database import db
from app.models.request.face_interface_req import PersonRecognizeRequest, BatchRecognizeRequest
from app.models.response.face_interface_rep import RecognizeResp, BBox, MatchItem, BatchRecognizeResp, FrameInfo
from app.models.api_response import StatusCode, ApiResponse
from app.core.logger import get_logger

logger = get_logger(__name__)

router = APIRouter(tags=["Face Recognition"])

BASE_DIR = Path(__file__).resolve().parent.parent
THRESHOLD = settings.face.threshold
CANDIDATE_THRESHOLD = settings.face.candidate_threshold
REC_MIN_FACE_HW = int(settings.face.rec_min_face_hw)

@router.post("/recognize2", response_model=ApiResponse)
async def recognize_face_api(request: PersonRecognizeRequest = Body(..., description="人脸识别请求")):
    import time
    t_total_start = time.time()

    logger.debug(f"[recognize] 接收到请求参数：{request}")

    # 1. 确定阈值
    threshold = request.threshold if request.threshold else THRESHOLD
    targets = request.targets if request.targets else []
    points = request.points if request.points else []

    # 2. 解析图片数据
    t_step = time.time()
    try:
        image_data, filename = await base64_to_mat(request.photo)
    except HTTPException as e:
        # base64 解码失败（HTTPException 由 base64_to_mat 抛出）
        logger.error(f"[recognize] 图片解析失败: {e.detail}")
        # 判断是 base64 解码错误还是图片格式错误
        if "base64" in str(e.detail).lower() or "decode" in str(e.detail).lower():
            return ApiResponse.error(
                status_code=StatusCode.BASE64_DECODE_ERROR,
                message=str(e.detail)
            )
        else:
            return ApiResponse.error(
                status_code=StatusCode.INVALID_IMAGE_FORMAT,
                message=str(e.detail)
            )
    logger.info(f"[recognize] [性能] 解析图片耗时: {(time.time()-t_step)*1000:.2f}ms")

    if image_data is None or not isinstance(image_data, np.ndarray) or image_data.size == 0:
        logger.error(f"[recognize] 未接收到有效图片数据或图像数据存在异常")
        return ApiResponse.error(
            status_code=StatusCode.INVALID_IMAGE_DATA,
            message="未接收到有效图片数据或图像数据存在异常"
        )

    # 3. 多人脸检测
    t_step = time.time()
    try:
        face_results = await ai_engine.detect_and_extract_all_faces(image_data)
    except Exception as e:
        logger.error(f"[recognize] 人脸检测服务内部错误: {e}")
        return ApiResponse.error(
            status_code=StatusCode.FACE_DETECTION_ERROR,
            message=f"人脸检测服务内部错误: {str(e)}"
        )
    logger.info(f"[性能] 多人脸检测总耗时(含进程通信): {(time.time()-t_step)*1000:.2f}ms")

    if not face_results:
        logger.info(f"[recognize] 未检测到有效人脸")
        logger.info(f"[recognize] [性能] 总请求耗时: {(time.time()-t_total_start)*1000:.2f}ms")
        return ApiResponse.error(
            status_code=StatusCode.NO_FACE_DETECTED,
            message="图像中未检测到人脸，请重新捕捉人脸"
        )

    logger.info(f"[recognize] 检测到 {len(face_results)} 张人脸")

    # 4. 过滤过小的人脸
    valid_faces = []
    for i, (face_image, bbox, tip) in enumerate(face_results):
        if face_image.shape[0] < REC_MIN_FACE_HW or face_image.shape[1] < REC_MIN_FACE_HW:
            logger.debug(f"[recognize] 人脸{i+1}过小: {face_image.shape[0]}x{face_image.shape[1]}px，跳过")
            continue
        valid_faces.append((face_image, bbox, tip))

    if not valid_faces:
        logger.info(f"[recognize] 所有检测到的人脸都过小，无法识别")
        return ApiResponse.error(
            status_code=StatusCode.FACE_TOO_SMALL,
            message="检测到的人脸像素过小，无法识别"
        )

    logger.info(f"[recognize] 有效人脸数量: {len(valid_faces)}")

    # 5. 预加载数据库数据
    all_docs = await cache_service.get_all_embeddings()

    if not all_docs:
        logger.info("[recognize] 数据库中没有有效人脸特征")
        return ApiResponse.error(
            status_code=StatusCode.DB_EMPTY,
            message="数据库为空，请先录入人员信息",
            data={
                "hasFace": True,
                "threshold": threshold,
                "match": [],
                "message": "数据库为空，请先录入人员信息"
            }
        )

    target_docs = []
    if targets:
        target_docs = await person.get_targets_embeddings(db, targets)
        logger.info(f"[recognize] targets 查询到 {len(target_docs)} 个候选人")

    # 6. 对每个人脸进行识别
    all_face_matches = []  # 存储所有人脸的匹配结果: [(similarity, doc, bbox, is_target), ...]

    for i, (face_image, bbox, tip) in enumerate(valid_faces):
        logger.info(f"[recognize] 处理人脸 {i+1}/{len(valid_faces)}")

        # 提取特征
        try:
            emb_q = await ai_engine.get_embedding(face_image)
        except Exception as e:
            logger.error(f"[recognize] 人脸{i+1}特征提取失败: {e}")
            continue

        # 双阶段匹配
        face_matches = []

        # 阶段1: targets 优先匹配（如果有targets）
        if target_docs:
            target_threshold = CANDIDATE_THRESHOLD  # 使用候选阈值
            target_results = ai_engine.find_top_matches(
                emb_q, target_docs, top_k=3, min_threshold=target_threshold
            )
            for sim, doc in target_results:
                face_matches.append((sim, doc, bbox, True))  # is_target=True
            logger.debug(f"[recognize] 人脸{i+1} targets匹配: {len(target_results)}个")

        # 阶段2: 全局匹配
        global_results = ai_engine.find_top_matches(
            emb_q, all_docs, top_k=3, min_threshold=threshold
        )
        for sim, doc in global_results:
            face_matches.append((sim, doc, bbox, False))  # is_target=False
        logger.debug(f"[recognize] 人脸{i+1} 全局匹配: {len(global_results)}个")

        # 将该人脸的匹配结果加入总列表
        all_face_matches.extend(face_matches)

    # 7. 全局去重和排序
    if not all_face_matches:
        logger.info("[recognize] 所有人脸都未找到匹配")
        return ApiResponse.error(
            status_code=StatusCode.NO_MATCH_FOUND,
            message="未找到匹配的人物（相似度低于阈值）",
            data={
                "hasFace": True,
                "threshold": threshold,
                "match": [],
                "message": "未找到匹配的人物（相似度低于阈值）"
            }
        )

    # 按相似度降序排序
    all_face_matches.sort(key=lambda x: x[0], reverse=True)

    # 去重：相同number的人物只保留相似度最高的
    seen_numbers = set()
    final_matches = []

    for sim, doc, bbox, is_target in all_face_matches:
        number = doc.get("number")
        if number and number not in seen_numbers:
            seen_numbers.add(number)
            final_matches.append((sim, doc, bbox, is_target))
            if len(final_matches) >= 3:  # 只取前3名
                break

    # 8. 构建响应
    match_items = []
    for sim, doc, bbox, is_target in final_matches:
        match_items.append(MatchItem(
            bbox=BBox(**bbox),
            id=str(doc["_id"]),
            name=doc.get("name"),
            number=doc.get("number"),
            similarity=f"{sim * 100:.2f}%",
            is_target=is_target
        ))

    # 9. 构建消息
    best_match = final_matches[0]
    best_name = best_match[1].get("name")
    best_number = best_match[1].get("number")

    message = f"匹配成功，≥阈值{threshold*100:.2f}%有{len(final_matches)}位，最相似的是{best_name}_{best_number}"

    logger.info(f"[recognize] [性能] 总请求耗时: {(time.time()-t_total_start)*1000:.2f}ms")

    return ApiResponse.success(
        message="识别成功",
        data={
            "hasFace": True,
            "threshold": threshold,
            "match": [item.model_dump() for item in match_items],
            "message": message
        }
    )