from pydantic import BaseModel
from typing import Optional
from bson import ObjectId

# MongoDB 使用 ObjectId 作为主键
class PersonBase(BaseModel):
    chinese_name: str
    description: Optional[str] = None

class Person(PersonBase):
    id: Optional[str]  # 使用字符串类型存储 MongoDB 的 ObjectId

    class Config:
        # 将 MongoDB 的 ObjectId 转换为字符串
        json_encoders = {
            ObjectId: str
        }

class PersonCreate(BaseModel):
    chinese_name: str
    description: Optional[str] = None
    photo_path: Optional[str] = None

# PersonRead 是返回给前端的模型
class PersonRead(BaseModel):
    id: str
    chinese_name: str
    description: Optional[str] = None
    photo_path: Optional[str] = None
