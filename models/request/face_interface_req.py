from pydantic import BaseModel
from app.models.schemas import PersonBase

"""
用于存放/recognize接口的请求模型
"""

class PersonRecognizeRequest(BaseModel):
    # 人脸recognize请求模型
    photo: str
    targets: list[PersonBase]
    threshold: float = None