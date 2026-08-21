from typing import Union

from pydantic import BaseModel


class GenerativeSessionParams(BaseModel):
    model_name: str
    task_name: str
    parameters: dict
    name: str
    description: Union[str, None] = None
    fine_tuning_run_id: Union[int, None] = None
