from pydantic import BaseModel, computed_field
from pathlib import Path
import tomli
from app.core.logger import get_logger

logger = get_logger(__name__)

config_path = Path(__file__).resolve().parent.parent / "config.toml"

class DBSettings(BaseModel):
    username: str
    password: str
    host: str
    port: str
    database: str
    auth_source: str
    limit: int = 10000 # 限制返回的结果条数

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

class ThreadSettings(BaseModel):
    max_workers: int

class GpuSettings(BaseModel):
    gpu_id: int

class FrontLoginSettings(BaseModel):
    username: str
    password: str


class LoggerSettings(BaseModel):
    level: str
    filename: str = "face_recognizer.log"

class FeatureImageSettings(BaseModel):
    max_feature_image_width_px: int = 720
    max_feature_image_height_px: int = 1280
    min_feature_image_width_px: int = 80
    min_feature_image_height_px: int= 80
    max_feature_image_size_m : int = 10
    max_face_hw: int = 300
    min_face_hw: int = 40

class Settings(BaseModel):
    db: DBSettings
    face: FaceSettings
    thread: ThreadSettings
    gpu: GpuSettings
    frontlogin: FrontLoginSettings
    feature_image: FeatureImageSettings
    logger: LoggerSettings

def load_config():
    logger.debug(f"Loading config from {config_path}")
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "rb") as f:
        config_data = tomli.load(f)

    logger.debug(f"Config loaded: {config_data}")

    return Settings(
        db=DBSettings(**config_data["db"]),
        face=FaceSettings(**config_data["face"]),
        thread=ThreadSettings(**config_data["threading"]),
        gpu=GpuSettings(**config_data["gpu"]),
        frontlogin=FrontLoginSettings(**config_data["frontlogin"]),
        feature_image=FeatureImageSettings(**config_data["image"]),
        logger=LoggerSettings(**config_data["logger"])
    )


settings = load_config()
