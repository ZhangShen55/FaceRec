"""
人脸特征缓存服务
负责人员信息在 Redis 中的缓存管理
"""
import json
import numpy as np
from typing import List, Dict, Optional
from app.core.redis_client import get_redis
from app.core.config import settings
from app.core.logger import get_logger
from app.services import person as person_service
from app.core.database import db

logger = get_logger(__name__)

# Redis Key 定义
EMBEDDINGS_KEY = "face:embeddings:all"  # 所有人员特征
LAST_REFRESH_KEY = "face:embeddings:last_refresh"  # 最后刷新时间


class CacheService:
    """缓存服务类"""

    @staticmethod
    async def reload_all_embeddings(force: bool = False) -> int:
        """
        从 MongoDB 重新加载所有人员特征到 Redis

        Args:
            force: 是否强制刷新（忽略缓存开关）

        Returns:
            加载的人员数量
        """
        if not settings.redis.cache.enable_embedding_cache and not force:
            logger.info("[Cache] 特征向量缓存未启用，跳过加载")
            return 0

        try:
            logger.info("[Cache] 开始加载所有人员特征到 Redis...")

            # 从 MongoDB 获取所有人员数据
            all_docs = await person_service.get_embeddings_for_match(db)

            if not all_docs:
                logger.warning("[Cache] MongoDB 中没有人员数据")
                return 0

            # 序列化数据
            redis_client = await get_redis()
            serialized_data = []

            for doc in all_docs:
                # 将 numpy array 转换为 list
                embedding = doc.get("embedding")
                if isinstance(embedding, np.ndarray):
                    embedding = embedding.tolist()
                elif isinstance(embedding, bytes):
                    # 如果是 Binary 类型，转换为 numpy 再转 list
                    embedding = np.frombuffer(embedding, dtype=np.float32).tolist()

                serialized_doc = {
                    "_id": str(doc["_id"]),
                    "name": doc.get("name"),
                    "number": doc.get("number"),
                    "photo_path": doc.get("photo_path"),
                    "embedding": embedding
                }
                serialized_data.append(serialized_doc)

            # 存储到 Redis
            json_data = json.dumps(serialized_data, ensure_ascii=False)

            # 使用 pipeline 提高性能
            async with redis_client.pipeline() as pipe:
                pipe.set(EMBEDDINGS_KEY, json_data)
                if settings.redis.cache.embeddings_ttl > 0:
                    pipe.expire(EMBEDDINGS_KEY, settings.redis.cache.embeddings_ttl)
                pipe.set(LAST_REFRESH_KEY, str(int(__import__('time').time())))
                await pipe.execute()

            logger.info(f"[Cache] ✅ 成功加载 {len(serialized_data)} 个人员特征到 Redis")
            return len(serialized_data)

        except Exception as e:
            logger.error(f"[Cache] ❌ 加载人员特征失败: {e}", exc_info=True)
            return 0

    @staticmethod
    async def get_all_embeddings() -> List[Dict]:
        """
        从 Redis 获取所有人员特征
        如果 Redis 中没有，则从 MongoDB 加载

        Returns:
            人员特征列表
        """
        if not settings.redis.cache.enable_embedding_cache:
            # 缓存未启用，直接从 MongoDB 读取
            logger.debug("[Cache] 缓存未启用，从 MongoDB 读取")
            return await person_service.get_embeddings_for_match(db)

        try:
            redis_client = await get_redis()
            cached_data = await redis_client.get(EMBEDDINGS_KEY)

            if cached_data:
                # 从 Redis 获取成功
                logger.debug("[Cache] 从 Redis 获取人员特征")
                serialized_data = json.loads(cached_data)

                # 将 list 转回 numpy array
                for doc in serialized_data:
                    if "embedding" in doc and isinstance(doc["embedding"], list):
                        doc["embedding"] = np.array(doc["embedding"], dtype=np.float32)

                return serialized_data
            else:
                # Redis 中没有数据，从 MongoDB 加载
                logger.warning("[Cache] Redis 中没有缓存，从 MongoDB 加载")
                await CacheService.reload_all_embeddings()

                # 再次尝试从 Redis 获取
                cached_data = await redis_client.get(EMBEDDINGS_KEY)
                if cached_data:
                    serialized_data = json.loads(cached_data)
                    for doc in serialized_data:
                        if "embedding" in doc and isinstance(doc["embedding"], list):
                            doc["embedding"] = np.array(doc["embedding"], dtype=np.float32)
                    return serialized_data
                else:
                    # 仍然失败，直接从 MongoDB 返回
                    return await person_service.get_embeddings_for_match(db)

        except Exception as e:
            logger.error(f"[Cache] 从 Redis 获取人员特征失败，降级到 MongoDB: {e}")
            return await person_service.get_embeddings_for_match(db)

    @staticmethod
    async def update_person_cache(person_dict: Dict) -> bool:
        """
        更新单个人员的缓存（增加或更新）

        Args:
            person_dict: 人员信息字典（包含 embedding）

        Returns:
            是否更新成功
        """
        if not settings.redis.cache.enable_embedding_cache or not settings.redis.cache.refresh_on_update:
            return True

        try:
            # 简单策略：直接重新加载全部数据
            # 对于 10000 人的规模，全量刷新也很快（<100ms）
            await CacheService.reload_all_embeddings()
            logger.info(f"[Cache] 人员 {person_dict.get('number')} 更新后已刷新缓存")
            return True

        except Exception as e:
            logger.error(f"[Cache] 更新人员缓存失败: {e}")
            return False

    @staticmethod
    async def delete_person_cache(number: str) -> bool:
        """
        删除人员后刷新缓存

        Args:
            number: 人员编号

        Returns:
            是否删除成功
        """
        if not settings.redis.cache.enable_embedding_cache or not settings.redis.cache.refresh_on_update:
            return True

        try:
            # 删除后重新加载全部数据
            await CacheService.reload_all_embeddings()
            logger.info(f"[Cache] 人员 {number} 删除后已刷新缓存")
            return True

        except Exception as e:
            logger.error(f"[Cache] 删除人员缓存失败: {e}")
            return False

    @staticmethod
    async def get_cache_info() -> Dict:
        """
        获取缓存信息（用于监控）

        Returns:
            缓存信息字典
        """
        try:
            redis_client = await get_redis()
            cached_data = await redis_client.get(EMBEDDINGS_KEY)
            last_refresh = await redis_client.get(LAST_REFRESH_KEY)

            if cached_data:
                data = json.loads(cached_data)
                count = len(data)
            else:
                count = 0

            return {
                "cached_count": count,
                "last_refresh": int(last_refresh) if last_refresh else None,
                "cache_enabled": settings.redis.cache.enable_embedding_cache,
                "refresh_on_startup": settings.redis.cache.refresh_on_startup,
                "refresh_interval_days": settings.redis.cache.refresh_interval_days,
                "refresh_on_update": settings.redis.cache.refresh_on_update
            }

        except Exception as e:
            logger.error(f"[Cache] 获取缓存信息失败: {e}")
            return {"error": str(e)}


# 导出单例
cache_service = CacheService()
