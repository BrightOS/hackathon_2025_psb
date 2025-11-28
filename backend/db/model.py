from pydantic import BaseModel

class DBEmailModel(BaseModel):
    sendler: str
    mail: str
    order: int
