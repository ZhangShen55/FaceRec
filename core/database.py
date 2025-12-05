# app/database.py
from motor.motor_asyncio import AsyncIOMotorClient
import os
from app.core.config import settings
from urllib.parse import quote_plus

MONGODB_URI = settings.db.url
MONGODB_DB = settings.db
if MONGODB_URI:
    # 例如: mongodb://user:pass@127.0.0.1:27017/facerecapi?authSource=admin
    client = AsyncIOMotorClient(MONGODB_URI)
    # 自动从 URI 里取库名；若 URI 没带路径，回退 MONGO_DB
    from pymongo.uri_parser import parse_uri
    parsed = parse_uri(MONGODB_URI)
    dbname = (parsed.get("database") or os.getenv("MONGO_DB", "facerecapi"))
else:
    # # 分段提供的情况：这里才进行一次编码（只编码一次！）
    # USER = quote_plus(os.getenv("MONGO_USER", "root"))
    # PASS = quote_plus(os.getenv("MONGO_PASS", "root"))
    # HOST = os.getenv("MONGO_HOST", "127.0.0.1")
    # PORT = os.getenv("MONGO_PORT", "27017")
    # DB   = os.getenv("MONGO_DB", "facerecapi")
    # AUTH = os.getenv("MONGO_AUTHSOURCE", DB)  # 用户建在哪个库，就写哪个；常见是 admin

    USER = quote_plus(MONGODB_DB.username)
    PASS = quote_plus(MONGODB_DB.password)
    HOST = MONGODB_DB.host
    PORT = MONGODB_DB.port
    DB   = MONGODB_DB.database
    AUTH = MONGODB_DB.auth_source  # admin


    uri = f"mongodb://{USER}:{PASS}@{HOST}:{PORT}/{DB}?authSource={AUTH}"
    client = AsyncIOMotorClient(uri)
    dbname = DB

db = client.get_database(dbname)

def get_session():
    return db
