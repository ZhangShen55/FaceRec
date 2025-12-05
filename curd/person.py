# app/person.py
from bson import ObjectId

# 获取单个人物
async def get_person(db, person_id: str):
    return await db["persons"].find_one({"_id": ObjectId(person_id)})

# 获取所有人物
async def get_persons(db, skip: int = 0, limit: int = 100):
    return await db["persons"].find().skip(skip).limit(limit).to_list(length=limit)

# 创建人物
async def create_person(db, person_dict: dict):
    res = await db["persons"].insert_one(person_dict)
    return await db["persons"].find_one({"_id": res.inserted_id}, {"embedding": 0})

# 删除人物
async def delete_person(db, person_id: str):
    result = await db["persons"].delete_one({"_id": ObjectId(person_id)})
    if result.deleted_count == 0:
        return None
    return {"id": person_id}


async def delete_persons_by_name(db, name_keyword: str):
    """
    根据姓名模糊删除人物
    """
    # MongoDB 使用正则表达式进行模糊查询
    result = await db["persons"].delete_many({"chinese_name": {"$regex": name_keyword, "$options": "i"}})
    return result.deleted_count

async def delete_person_by_id(db, person_id: str):
    """
    根据ID删除人物
    """
    result = await db["persons"].delete_one({"_id": ObjectId(person_id)})
    if result.deleted_count == 0:
        return None
    return {"id": person_id}


async def delete_persons_by_name_exact(db, name: str):
    """
    按人物姓名精确匹配删除
    """
    result = await db["persons"].delete_many({"chinese_name": name})
    return result.deleted_count


async def get_persons_by_name(db, name_keyword: str):
    """
    根据姓名模糊查询人物
    """
    persons = await db["persons"].find({"chinese_name": {"$regex": name_keyword, "$options": "i"}}).to_list(length=100)
    return persons


async def get_person_by_id(db, person_id: str):
    """
    根据ID精确查询人物
    """
    person = await db["persons"].find_one({"_id": ObjectId(person_id)})
    return person


async def get_embeddings_for_match(db, limit=7000):
    # 只投影需要的字段，减少 IO
    cursor = db["persons"].find(
        {}, {"embedding": 1, "chinese_name": 1, "photo_path": 1}
    )
    return await cursor.limit(limit).to_list(length=limit)