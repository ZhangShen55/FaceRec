"""
Redis 连接管理
提供 Redis 连接池和基础操作
"""
from typing import Optional
import redis.asyncio as aioredis
from redis.asyncio.connection import ConnectionPool
from app.core.config import settings
from app.core.logger import get_logger

logger = get_logger(__name__)


class RedisClient:
    """Redis 客户端单例"""

    _pool: Optional[ConnectionPool] = None
    _client: Optional[aioredis.Redis] = None

    @classmethod
    async def get_pool(cls) -> ConnectionPool:
        """获取连接池"""
        if cls._pool is None:
            logger.info(f"[Redis] 创建连接池: {settings.redis.host}:{settings.redis.port}")
            cls._pool = ConnectionPool(
                host=settings.redis.host,
                port=settings.redis.port,
                password=settings.redis.password if settings.redis.password else None,
                db=settings.redis.db,
                max_connections=settings.redis.max_connections,
                socket_timeout=settings.redis.socket_timeout,
                socket_connect_timeout=settings.redis.socket_connect_timeout,
                decode_responses=settings.redis.decode_responses
            )
        return cls._pool

    @classmethod
    async def get_client(cls) -> aioredis.Redis:
        """获取 Redis 客户端"""
        if cls._client is None:
            pool = await cls.get_pool()
            cls._client = aioredis.Redis(connection_pool=pool)
            logger.info("[Redis] 客户端已创建")
        return cls._client

    @classmethod
    async def close(cls):
        """关闭连接"""
        if cls._client:
            await cls._client.close()
            logger.info("[Redis] 客户端已关闭")
            cls._client = None
        if cls._pool:
            await cls._pool.disconnect()
            logger.info("[Redis] 连接池已关闭")
            cls._pool = None

    @classmethod
    async def ping(cls) -> bool:
        """测试连接"""
        try:
            client = await cls.get_client()
            await client.ping()
            logger.info("[Redis] 连接测试成功")
            return True
        except Exception as e:
            logger.error(f"[Redis] 连接测试失败: {e}")
            return False


# 全局 Redis 客户端获取函数
async def get_redis() -> aioredis.Redis:
    """获取 Redis 客户端实例"""
    return await RedisClient.get_client()
