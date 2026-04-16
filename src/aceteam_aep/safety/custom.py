import uuid

from pydantic import BaseModel, Field


class CustomPolicy(BaseModel):
    """Custom policy for the safety detector."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    rule: str
    enabled: bool


__all__ = [
    "CustomPolicy",
]
