import logging
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from schemas import RowScheme, TokenScheme
from database import get_db_session
from models import Finance_data


finance_router = APIRouter()

logger = logging.getLogger(__name__)

#CRUD
# check if filter_by is needed
"""
Maybe it's worth changing ** -> *

all_filters = [UserModal.role == 'admin']

if user.last_name:
    all_filters.append(UserModal.last_name == 'Deo')

db.session.query(
    UserModal.username
).filter(
    **all_filters
).all()

"""
def get_finance_data_by_params(db: Session, symbol: str, gecko_id: str, llama_id: str,
                       category: str, chain: str, start_date: str, end_date: str):
    filters = []
    
    if symbol != "None":
        filters.append(Finance_data.symbol == symbol)
    if gecko_id != "None":
        filters.append(Finance_data.gecko_id == gecko_id)
    if llama_id != "None":
        filters.append(Finance_data.llama_id == llama_id)
    if category != "None":
        filters.append(Finance_data.category == category)
    if chain != "None":
        filters.append(Finance_data.chain == chain)
    if start_date != "None":
        filters.append(Finance_data.date >= start_date)
    if end_date != "None":
        filters.append(Finance_data.date <= end_date)
    
    if len(filters) != 0:
        return db.query(Finance_data).filter(*filters).all()

    return db.query(Finance_data).all()

def get_tokens(db: Session, chain: str = "Ethereum"):
    return db.query(Finance_data.address).distinct(Finance_data.address).all()



@finance_router.get("/", response_model=list[RowScheme])
def api_data(skip: int = 0, limit: int = 100, symbol: str = "None", gecko_id: str = "None",
             llama_id: str = "None", category: str = "None", chain: str = "None",
             start_date: str = "None", end_date: str = "None", db: Session = Depends(get_db_session)):

    data = get_finance_data_by_params(db, symbol, gecko_id, llama_id, category,
                                      chain, start_date, end_date)
    return data

@finance_router.get("/tokens", response_model=list[TokenScheme])
def api_tokens(db: Session = Depends(get_db_session)):

    data = get_tokens(db)
    return data




