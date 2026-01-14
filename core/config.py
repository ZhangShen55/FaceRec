from pydantic import BaseModel, computed_field
from pathlib import Path
import tomli
import logging

logger = logging.getLogger(__name__)
logger.level = logging.DEBUG

config_path = Path(__file__).resolve().parent.parent / "config.toml"

class DBSettings(BaseModel):
    username: str
    password: str
    host: str
    port: str
    database: str
    auth_source: str
    limit: int = 10000

    @computed_field
    def url(self) -> str:
        # 构建 URL mongodb://user:pass@host:port/db?authSource=admin
        return (
            f"mongodb://{self.username}:{self.password}@"
            f"{self.host}:{self.port}/{self.database}"
            f"?authSource={self.auth_source}"
        )
class FaceSettings(BaseModel):
    threshold: float
    candidate_threshold: float
    rec_min_face_hw: int

class ThreadSettings(BaseModel):
    max_workers: int

class GpuSettings(BaseModel):
    gpu_id: int

class FrontLoginSettings(BaseModel):
    username: str
    password: str

class LoggerSettings(BaseModel):
    level: str = "INFO"
    log_path: str = "/app/logs/facerec.log"

class FeatureImageSettings(BaseModel):
    max_feature_image_width_px: int = 720
    max_feature_image_height_px: int = 1280
    min_feature_image_width_px: int = 80
    min_feature_image_height_px: int= 80
    max_feature_image_size_m : int = 10
    max_face_hw: int = 300
    min_face_hw: int = 40

class StatsSettings(BaseModel):
    retention_days: int = 7  # 详细日志保留天数
    hourly_retention_days: int = 30  # 按小时聚合数据保留天数

class RedisCacheSettings(BaseModel):
    """Redis 缓存配置"""
    embeddings_ttl: int = 0  # 人脸特征向量缓存过期时间（0表示永不过期）
    enable_embedding_cache: bool = True  # 是否启用特征向量缓存
    refresh_on_startup: bool = True  # 启动时是否自动加载
    refresh_interval_days: int = 7  # 定时全量刷新间隔（天）
    refresh_on_update: bool = True  # 人员增删改时是否实时更新

class RedisSettings(BaseModel):
    """Redis 配置"""
    host: str = "localhost"
    port: int = 6379
    password: str = ""
    db: int = 0
    max_connections: int = 50
    socket_timeout: int = 5
    socket_connect_timeout: int = 5
    decode_responses: bool = True
    cache: RedisCacheSettings = RedisCacheSettings()

    @computed_field
    def url(self) -> str:
        """构建 Redis URL"""
        if self.password:
            return f"redis://:{self.password}@{self.host}:{self.port}/{self.db}"
        return f"redis://{self.host}:{self.port}/{self.db}"

class Settings(BaseModel):
    db: DBSettings
    face: FaceSettings
    thread: ThreadSettings
    gpu: GpuSettings
    frontlogin: FrontLoginSettings
    feature_image: FeatureImageSettings
    logger: LoggerSettings
    stats: StatsSettings
    redis: RedisSettings

def load_config():
    logger.debug(f"Loading config from {config_path}")
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "rb") as f:
        config_data = tomli.load(f)

    logger.debug(f"Config loaded: {config_data}")

    # 处理 Redis 配置（包含嵌套的 cache 配置）
    redis_config = config_data.get("redis", {})
    if "cache" in redis_config:
        redis_cache = RedisCacheSettings(**redis_config["cache"])
        redis_config["cache"] = redis_cache

    return Settings(
        db=DBSettings(**config_data["db"]),
        face=FaceSettings(**config_data["face"]),
        thread=ThreadSettings(**config_data["threading"]),
        gpu=GpuSettings(**config_data["gpu"]),
        frontlogin=FrontLoginSettings(**config_data["frontlogin"]),
        feature_image=FeatureImageSettings(**config_data["image"]),
        logger=LoggerSettings(**config_data["logger"]),
        stats=StatsSettings(**config_data.get("stats", {})),  # 兼容旧配置
        redis=RedisSettings(**redis_config) if redis_config else RedisSettings()  # 使用默认值
    )


settings = load_config()
