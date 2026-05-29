from pydantic import BaseModel, Field

class Entity(BaseModel):
    name: str = Field(..., description="Unique name of the entity")
    join_key: str = Field(..., description="The primary join column name")
    description: str = Field("", description="A description of the entity context")

    def __hash__(self):
        return hash((self.name, self.join_key))

    def __eq__(self, other):
        if not isinstance(other, Entity):
            return False
        return self.name == other.name and self.join_key == other.join_key
