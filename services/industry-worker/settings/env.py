from pydantic import BaseSettings

class Settings(BaseSettings):
    username: str
    password: str
    database: str
    host: str

    finance_data_delay: float = 86400 # 24 hours
    
    initialize: str = "True"


env = Settings()
