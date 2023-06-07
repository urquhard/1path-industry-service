from typing import Optional

from pydantic import BaseSettings


class Settings(BaseSettings):
    openapi_url: Optional[str] = None
    username: str
    password: str
    host: str
    database: str


settings = Settings()

