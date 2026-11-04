from pydantic import BaseModel
from typing import Any


class TypeInfo(BaseModel):
    type: str
    items: "TypeInfo | None" = None
    properties: "dict[str, TypeInfo] | None" = None


class FunctionDefinition(BaseModel):
    name: str
    description: str
    parameters: dict[str, TypeInfo]
    returns: TypeInfo


class TestCase(BaseModel):
    prompt: str


class FunctionCallResult(BaseModel):
    prompt: str
    name: str
    parameters: dict[str, Any]


class DecoderSettings(BaseModel):
    verbose: bool = False
    model_name: str = "Qwen/Qwen3-0.6B"
