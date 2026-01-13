"""
统一 API 响应格式模型
所有接口 HTTP 状态码永远是 200，通过 status_code 字段区分成功/失败
"""
from typing import Optional, Any
from pydantic import BaseModel
from enum import IntEnum


class StatusCode(IntEnum):
    """统一状态码枚举"""
    # 成功
    SUCCESS = 200

    # 客户端错误 4xx
    PARTIAL_SUCCESS = 207       # 部分成功（批量操作时部分失败）
    BAD_REQUEST = 400           # 请求参数错误
    NOT_FOUND = 404             # 资源未找到
    UNPROCESSABLE_ENTITY = 422  # 无法处理的实体（人脸检测失败等）
    FACE_TOO_SMALL = 423        # 人脸尺寸过小

    # 服务器错误 5xx
    INTERNAL_ERROR = 500        # 服务器内部错误（数据库操作失败等）
    FACE_DETECTION_ERROR = 501  # 人脸检测服务内部错误
    FEATURE_EXTRACT_ERROR = 502 # 特征提取失败
    FILE_SAVE_ERROR = 503       # 文件保存失败


class ApiResponse(BaseModel):
    """统一 API 响应格式 - 所有接口 HTTP 状态码都是 200"""
    status_code: int
    message: str
    data: Optional[Any] = None

    @classmethod
    def success(cls, data: Any = None, message: str = "操作成功"):
        """成功响应"""
        return cls(status_code=StatusCode.SUCCESS, message=message, data=data)

    @classmethod
    def error(cls, status_code: int, message: str, data: Any = None):
        """错误响应"""
        return cls(status_code=status_code, message=message, data=data)
