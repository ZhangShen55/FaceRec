"""
人脸特征缓存服务
负责人员信息在 Redis 中的缓存管理
"""
import json
import pickle
import numpy as np
from typing import List, Dict, Optional
from app.core.redis_client import get_redis
from app.core.config import settings
from app.core.logger import get_logger
from app.services import person as person_service
from app.core.database import db

logger = get_logger(__name__)

# 序列化格式版本（用于兼容性检查和迁移）
CACHE_VERSION = 2  # 1=JSON, 2=Pickle
VERSION_KEY = "face:embeddings:version"

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

            # 序列化数据（使用 Pickle，性能比 JSON 快 5-10 倍）
            redis_client = await get_redis()
            serialized_data = []

            for doc in all_docs:
                # 处理 embedding：统一转换为 numpy array
                embedding = doc.get("embedding")
                if isinstance(embedding, np.ndarray):
                    embedding = embedding.astype(np.float32)
                elif isinstance(embedding, bytes):
                    # 如果是 Binary 类型，转换为 numpy
                    embedding = np.frombuffer(embedding, dtype=np.float32)
                elif isinstance(embedding, list):
                    embedding = np.array(embedding, dtype=np.float32)
                else:
                    continue

                # 保持 numpy array 格式（Pickle 可以直接序列化 numpy）
                serialized_doc = {
                    "_id": str(doc["_id"]),
                    "name": doc.get("name"),
                    "number": doc.get("number"),
                    "photo_path": doc.get("photo_path"),
                    "embedding": embedding  # numpy array，Pickle 可以直接序列化
                }
                serialized_data.append(serialized_doc)

            # 使用 Pickle 序列化（比 JSON 快 5-10 倍，数据小 50%）
            pickle_data = pickle.dumps(serialized_data, protocol=pickle.HIGHEST_PROTOCOL)

            # 使用 pipeline 提高性能
            # 注意：decode_responses=False 时，字符串值需要编码为 bytes
            async with redis_client.pipeline() as pipe:
                pipe.set(EMBEDDINGS_KEY, pickle_data)  # pickle_data 已经是 bytes
                if settings.redis.cache.embeddings_ttl > 0:
                    pipe.expire(EMBEDDINGS_KEY, settings.redis.cache.embeddings_ttl)
                # Redis 客户端会自动将字符串编码为 bytes（当 decode_responses=False 时）
                pipe.set(LAST_REFRESH_KEY, str(int(__import__('time').time())))
                pipe.set(VERSION_KEY, str(CACHE_VERSION))  # 记录版本号
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
                
                # 检查版本号，判断序列化格式
                version = await redis_client.get(VERSION_KEY)
                version = int(version.decode('utf-8')) if version else None  # None 表示未设置版本号
                
                # 智能检测：先尝试 Pickle（如果版本 >= 2 或版本号未设置但数据看起来像 Pickle）
                # Pickle 数据通常以特定字节开头（如 \x80, \x95 等）
                is_likely_pickle = version is None or version >= 2
                
                if is_likely_pickle:
                    # 尝试 Pickle 反序列化
                    try:
                        serialized_data = pickle.loads(cached_data)
                        # Pickle 反序列化后，embedding 已经是 numpy array，无需转换
                        # 如果版本号未设置，更新版本号
                        if version is None:
                            await redis_client.set(VERSION_KEY, str(CACHE_VERSION))
                        return serialized_data
                    except Exception as pickle_error:
                        # Pickle 失败，可能是旧版 JSON 数据
                        logger.debug(f"[Cache] Pickle 反序列化失败，尝试 JSON: {pickle_error}")
                        # 检查是否是有效的 JSON（JSON 必须是 UTF-8 字符串）
                        try:
                            # 尝试解码为字符串（decode_responses=False 时，cached_data 是 bytes）
                            cached_str = cached_data.decode('utf-8') if isinstance(cached_data, bytes) else cached_data
                            # 尝试 JSON 解析
                            serialized_data = json.loads(cached_str)
                            for doc in serialized_data:
                                if "embedding" in doc and isinstance(doc["embedding"], list):
                                    doc["embedding"] = np.array(doc["embedding"], dtype=np.float32)
                            # 自动迁移到 Pickle 格式
                            logger.info("[Cache] 检测到旧版本 JSON 格式，自动迁移到 Pickle")
                            await CacheService.reload_all_embeddings()
                            cached_data = await redis_client.get(EMBEDDINGS_KEY)
                            if cached_data:
                                return pickle.loads(cached_data)
                            return serialized_data
                        except (UnicodeDecodeError, json.JSONDecodeError) as json_error:
                            # JSON 解析也失败，重新加载
                            logger.warning(f"[Cache] JSON 反序列化也失败: {json_error}，重新加载")
                            await CacheService.reload_all_embeddings()
                            cached_data = await redis_client.get(EMBEDDINGS_KEY)
                            if cached_data:
                                return pickle.loads(cached_data)
                else:
                    # 版本号明确是 1，使用 JSON
                    try:
                        # decode_responses=False 时，cached_data 是 bytes
                        cached_str = cached_data.decode('utf-8') if isinstance(cached_data, bytes) else cached_data
                        serialized_data = json.loads(cached_str)
                        for doc in serialized_data:
                            if "embedding" in doc and isinstance(doc["embedding"], list):
                                doc["embedding"] = np.array(doc["embedding"], dtype=np.float32)
                        # 自动迁移到 Pickle 格式
                        logger.info("[Cache] 检测到旧版本 JSON 格式，自动迁移到 Pickle")
                        await CacheService.reload_all_embeddings()
                        cached_data = await redis_client.get(EMBEDDINGS_KEY)
                        if cached_data:
                            return pickle.loads(cached_data)
                        return serialized_data
                    except Exception as json_error:
                        logger.warning(f"[Cache] JSON 反序列化失败: {json_error}，尝试 Pickle")
                        # JSON 失败，尝试 Pickle（可能是版本号错误）
                        try:
                            serialized_data = pickle.loads(cached_data)
                            # 更新版本号
                            await redis_client.set(VERSION_KEY, str(CACHE_VERSION))
                            return serialized_data
                        except:
                            # 都失败，重新加载
                            logger.warning("[Cache] 所有反序列化方法都失败，重新加载")
                            await CacheService.reload_all_embeddings()
                            cached_data = await redis_client.get(EMBEDDINGS_KEY)
                            if cached_data:
                                return pickle.loads(cached_data)
            else:
                # Redis 中没有数据，从 MongoDB 加载
                logger.warning("[Cache] Redis 中没有缓存，从 MongoDB 加载")
                await CacheService.reload_all_embeddings()

                # 再次尝试从 Redis 获取
                cached_data = await redis_client.get(EMBEDDINGS_KEY)
                if cached_data:
                    return pickle.loads(cached_data)
                else:
                    # 仍然失败，直接从 MongoDB 返回
                    return await person_service.get_embeddings_for_match(db)

        except Exception as e:
            logger.error(f"[Cache] 从 Redis 获取人员特征失败，降级到 MongoDB: {e}", exc_info=True)
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
            version = await redis_client.get(VERSION_KEY)

            count = 0
            cache_format = "unknown"
            if cached_data:
                try:
                    version_int = int(version.decode('utf-8')) if version else 1
                    if version_int >= 2:
                        # Pickle 格式
                        data = pickle.loads(cached_data)
                        count = len(data)
                        cache_format = "pickle"
                    else:
                        # JSON 格式（兼容）
                        cached_str = cached_data.decode('utf-8') if isinstance(cached_data, bytes) else cached_data
                        data = json.loads(cached_str)
                        count = len(data)
                        cache_format = "json"
                except Exception as e:
                    logger.warning(f"[Cache] 解析缓存数据失败: {e}")

            return {
                "cached_count": count,
                "last_refresh": int(last_refresh.decode('utf-8')) if last_refresh else None,
                "cache_format": cache_format,
                "cache_version": int(version.decode('utf-8')) if version else 1,
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
