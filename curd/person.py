# app/person.py
from bson import ObjectId
from typing import List
from app.core.config import settings
from app.core.exceptions import DatabaseError
from app.schemas.schemas import PersonBase

# 获取单个人物
async def get_person(db, person_id: str):
    return await db["persons"].find_one({"_id": ObjectId(person_id)})

# 获取所有人物
async def get_persons(db, skip: int = 0, limit: int = 100):
    return await db["persons"].find().skip(skip).limit(limit).to_list(length=limit)


# 创建人物
async def create_person(db, person_dict: dict):

    # 唯一性校验
    exists = await db.persons.find_one(
        {"name": person_dict["name"], "number": person_dict["number"]}
    )

    if exists:
        raise DatabaseError(status_code=400, detail=f"请勿重复创建人物,name: {person_dict['name']}, number: {person_dict['number']}")
    # 插入
    res = await db["persons"].insert_one(person_dict)
    return await db["persons"].find_one({"_id": res.inserted_id}, {"embedding": 0})

# 删除人物
async def delete_person(db, person_id: str):
    result = await db["persons"].delete_one({"_id": ObjectId(person_id)})
    if result.deleted_count == 0:
        return None
    return {"_id": person_id}


async def delete_persons_by_name(db, name_keyword: str):
    """
    根据姓名模糊删除人物
    """
    # MongoDB 使用正则表达式进行模糊查询
    result = await db["persons"].delete_many({"name": {"$regex": name_keyword, "$options": "i"}})
    return result.deleted_count

async def delete_person_by_id(db, person_id: str):
    """
    根据ID删除人物
    """
    result = await db["persons"].delete_one({"_id": ObjectId(person_id)})
    if result.deleted_count == 0:
        return None
    return {"_id": person_id}


async def delete_persons_by_name_exact(db, name: str):
    """
    按人物姓名精确匹配删除
    """
    result = await db["persons"].delete_many({"name": name})
    return result.deleted_count


async def get_persons_by_name(db, name_keyword: str):
    """
    根据姓名模糊查询人物
    """
    persons = await db["persons"].find({"name": {"$regex": name_keyword, "$options": "i"}}).to_list(length=100)
    if not persons:
        return None
    return persons


async def get_person_by_id(db, person_id: str):
    """
    根据ID精确查询人物
    """
    person = await db["persons"].find_one({"_id": ObjectId(person_id)})
    return person

async def get_person_by_number(db, number: str):
    """
    根据 number 精确查询人物
    :param db: Motor 数据库对象
    :param number: 人物编号（字符串）
    :return: 查到则返回 dict；未查到返回 None（或按需抛 404）
    """
    person = await db["persons"].find_one({"number": number})
    if not person:
        return None
    return person


async def get_persons_by_name_and_number(db, name: str, number: str):
    """
    根据 name + number 组合精确查询人物(1条数据)
    """
    person = await db["persons"].find_one(
        {"name": {"$regex": name, "$options": "i"}, "number": number}
    )
    return person

LIMIT = settings.db.limit
async def get_embeddings_for_match(db, limit=LIMIT):
    cursor = db["persons"].find(
        {}, {"embedding": 1, "name": 1, "photo_path": 1}
    )
    return await cursor.limit(limit).to_list(length=limit)


# 通过排课人物候选表
async def get_persons_embeddings(db, persons: List[PersonBase]) -> List[dict]:
    """
    通过排课人物候选列表，获取人物embedding列表
    """
    if not persons:
        return []
        # 构建 $or 查询条件
        # 注意：这里假设数据库字段已同步修改为 name 和 number
        # 如果数据库还没改名，需要映射 {"name": c.name, "number": c.number}
    query_conditions = []
    for p in persons:
        condition = {"name": p.name}
        if p.number:
            condition["number"] = p.number
        query_conditions.append(condition)

    print(f"query_conditions======="
          f": {query_conditions}")
    if not query_conditions:
        return []

    projection = {"embedding": 1, "name": 1, "number": 1, "photo_path": 1}
    cursor = db["persons"].find({"$or": query_conditions}, projection)

    return await cursor.to_list(length=None)