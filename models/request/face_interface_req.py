from pydantic import BaseModel
from app.models.schemas import PersonBase
from typing import Optional, List

"""
用于存放/recognize接口的请求模型
"""
class Point(BaseModel):
    x: int
    y: int

class PersonRecognizeRequest(BaseModel):
    # 人脸recognize请求模型
    photo: str
    points: List[dict[Point]] = None # 可传递划区域多点，只识别区域内的人脸
    targets: Optional[List[str]] = None
    threshold: Optional[float] = None

class BatchRecognizeRequest(BaseModel):
    """
    批量识别请求（单人多帧）
    用于通过多张图片提高识别准确率
    """
    photos: List[str]  # Base64 编码的图片列表（单人多帧）
    targets: Optional[List[str]] = None  # 可选的候选人编号列表
    threshold: Optional[float] = None  # 可选的阈值
