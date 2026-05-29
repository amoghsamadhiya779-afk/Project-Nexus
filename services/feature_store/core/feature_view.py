from datetime import timedelta
from typing import List, Dict, Callable, Optional
from pydantic import BaseModel, Field, ConfigDict
from services.feature_store.core.entity import Entity
from services.feature_store.core.source import BatchSource, StreamSource

class Feature(BaseModel):
    name: str
    value_type: str = "float"
    description: str = ""

class FeatureView(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    name: str
    entities: List[Entity]
    features: List[Feature]
    batch_source: BatchSource
    stream_source: Optional[StreamSource] = None
    ttl: timedelta = Field(default=timedelta(days=365))
    transformation: Optional[Callable] = None

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        if not isinstance(other, FeatureView):
            return False
        return self.name == other.name

class FeatureRegistry(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    feature_views: Dict[str, FeatureView] = Field(default_factory=dict)
    entities: Dict[str, Entity] = Field(default_factory=dict)

    def register_entity(self, entity: Entity) -> None:
        self.entities[entity.name] = entity

    def register_feature_view(self, fv: FeatureView) -> None:
        self.feature_views[fv.name] = fv
        for entity in fv.entities:
            self.register_entity(entity)
