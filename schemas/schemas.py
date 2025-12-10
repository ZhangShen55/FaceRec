from pydantic import BaseModel
from typing import Optional
from pydantic import Field
from bson import ObjectId

class PersonBase(BaseModel):
    name: str = None # 姓名
    number: Optional[str] = None # 编号

class Person(PersonBase):
    id: Optional[str]  # 使用字符串类型存储 MongoDB 的 ObjectId
    class Config:
        json_encoders = {
            ObjectId: str
        }

class PersonCreate(BaseModel):
    name: str
    number: Optional[str] = None
    photo_path: Optional[str] = None

# PersonRead 是返回给前端的模型
class PersonRead(BaseModel):
    id: str
    name: str
    number: Optional[str] = None
    photo_path: Optional[str] = None


class PersonRecognizeRequest(BaseModel):
    photo: str
    targets: list[PersonBase]