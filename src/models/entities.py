from pydantic import BaseModel, Field
from typing import List


class Entities(BaseModel):

    names: List[str] = Field(
        ...,
        description="All the person, organization, or business entities that "
        "appear in the text",
    )
