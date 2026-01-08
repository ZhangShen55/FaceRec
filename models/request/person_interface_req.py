from pydantic import BaseModel
from app.models.schemas import PersonBase
from typing import Optional

"""
用于存放/persons接口的请求模型
"""

class PersonFeatureRequest(BaseModel):
    # 人脸特征请求
    photo: str
    name: str
    number: str

class PersonsFeatureRequest(BaseModel):
    persons: list[PersonFeatureRequest]


class GetPersonListRequest(BaseModel):
    skip: int = 0 # 跳过数据
    limit: int = 100 # 默认返回100条数据


class SearchPersonRequest(BaseModel):
    name: Optional[str] = None
    number: Optional[str] = None


class DeletePersonRequest(BaseModel):
    id: str = None
    name: str = None
    number: str = None


class DeletePersonByIdRequest(BaseModel):
    """
    按ID删除人物请求模型
    - ID是唯一标识，不会出现误删
    """
    id: str

