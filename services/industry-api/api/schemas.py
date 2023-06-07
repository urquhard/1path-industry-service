from datetime import datetime
from enum import Enum
from typing import List, Literal, Optional, Union

from pydantic import BaseModel, Field



class RowScheme(BaseModel):
    index: int
    date: str
    symbol: str
    gecko_id: str
    llama_id: str
    category: str
    chain: str
    address: Optional[str]
    price: Optional[float]
    market_cap: Optional[float]
    volume: Optional[float]
    TVL: Optional[float]
    
    class Config:
        orm_mode = True


class FinanceScheme(BaseModel):
    data: List[RowScheme]
    
    class Config:
        orm_mode = True

class TokenScheme(BaseModel):
    address: Optional[str]
    class Config:
        orm_mode = True

