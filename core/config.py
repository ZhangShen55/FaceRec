from pydantic import BaseModel, computed_field
from pathlib import Path
import tomli
from app.core.logger import get_logger

logger = get_logger(__name__)

class DBSettings(BaseModel):
    username: str
    password: str
    host: str
    port: str
    database: str
    auth_source: str

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

class Settings(BaseModel):
    db: DBSettings
    face: FaceSettings
    thread: ThreadSettings
    gpu: GpuSettings
    frontlogin: FrontLoginSettings

def load_config():
    config_path = Path(__file__).resolve().parent.parent / "config.toml"
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
        frontlogin=FrontLoginSettings(**config_data["frontlogin"])
    )

settings = load_config()
