from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    HOST: str = "localhost"
    PORT: str = "8000"
    MODEL_FOLDER_ID: str = ""
    MODEL_API_ID: str = ""
    MODEL_API_KEY: str = ""
    MONGO_HOST: str = ""
    MONGO_PORT: str = ""
    MONGO_USER: str = ""
    MONGO_PASS: str = ""

    class Config:
        env_file = ".env"

settings = Settings()
