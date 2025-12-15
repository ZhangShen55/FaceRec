from pydantic import BaseModel
from app.models.schemas import PersonBase

"""
用于存放/persons接口的请求模型
"""

class PersonFeatureRequest(BaseModel):
    # 人脸特征请求
    photo: str
    name: str
    number: str


class GetPersonListRequest(BaseModel):
    skip: int = 0 # 跳过数据
    limit: int = 100 # 默认返回100条数据


class SearchPersonRequest(BaseModel):
    name: str
    number: str


class DeletePersonRequest(BaseModel):
    id: str = None
    name: str = None
    number: str = None
