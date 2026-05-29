import os
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    PROJECT_NAME: str = "nexus"
    ENV: str = "development"
    
    # Cache & Online Storage
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    
    # Databases
    POSTGRES_HOST: str = "localhost"
    POSTGRES_PORT: int = 5432
    POSTGRES_DB: str = "nexus"
    POSTGRES_USER: str = "nexus"
    POSTGRES_PASSWORD: str = "nexus_secure_pass"
    
    # Message Broker
    KAFKA_BOOTSTRAP_SERVERS: str = "localhost:9092"
    
    # Model Registry
    MLFLOW_TRACKING_URI: str = "http://localhost:5000"
    
    # Cross-Platform Local Storage Paths
    BASE_DATA_DIR: str = "C:/data" if os.name == "nt" else str(Path.home() / "data")

    class Config:
        env_file = ".env"
        extra = "ignore"

config = Settings()
