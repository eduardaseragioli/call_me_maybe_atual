from pydantic import BaseModel, ValidationError
from typing import Any, Dict

class TypeInfo(BaseModel):
    type: str

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
    parameters: Dict[str, Any]