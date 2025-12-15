from typing import Optional

from pydantic import BaseModel
from app.models.schemas import PersonBase

"""
用于存放/persons接口的响应模型

       "id" : str(doc["_id"]),
        "name": doc.get("name"),
        "number": doc.get("number"),
        "photo_path": doc.get("photo_path"),
        "tip": tip
"""

class PersonFeatureResponse(BaseModel):
    # 人脸特征响应
    id: str
    name: str
    number: str
    photo_path: str
    tip: str


# PersonRead 是返回给前端的模型
class PersonRead(BaseModel):
    id: str
    name: str
    number: Optional[str] = None
    photo_path: Optional[str] = None
