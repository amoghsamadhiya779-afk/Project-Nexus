from pydantic import BaseModel, Field
from typing import Optional

class BatchSource(BaseModel):
    type: str = Field("parquet", description="Database source class")
    path: Optional[str] = Field(None, description="Local or cloud folder path")
    connection_string: Optional[str] = None
    table_name: Optional[str] = None
    timestamp_field: str = "timestamp"

class StreamSource(BaseModel):
    type: str = "kafka"
    bootstrap_servers: str = "localhost:9092"
    topic: str
