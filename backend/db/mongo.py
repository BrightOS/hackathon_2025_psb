from pymongo import MongoClient

from config import settings
from db.model import DBEmailModel


class MongoManager:
    def __init__(self):
        self.client = MongoClient(
            f'mongodb://{settings.MONGO_USER}:{settings.MONGO_PASS}@{settings.MONGO_HOST}:{settings.MONGO_PORT}/'
        )

        self.db = self.client['service']
        self.collection = self.db['mails']

    def set_mail(self, model: DBEmailModel):
        result = self.collection.insert_one(model)
        return result.inserted_id
    
    def get_mail(self, sendler: str):
        return self.collection.find({"sendler": sendler})
