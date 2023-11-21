import logging
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from database import get_db_session
from models import IndustryReturns, PValueData, ResTest, SE, VifValues, XDict


model_router = APIRouter()

logger = logging.getLogger(__name__)

def get_model_data_by_date(db: Session, table, start_date: str, end_date: str):
    filters = []
    
    if start_date != "None":
        filters.append(table.index >= start_date)
    if end_date != "None":
        filters.append(table.index <= end_date)
    
    if len(filters) != 0:
        return db.query(table).filter(*filters).all()

    return db.query(table).all()


@model_router.get("/{table_name}")
def api_data(table_name: str, start_date: str = "None",
             end_date: str = "None", db: Session = Depends(get_db_session)):
    if table_name == "IndustryReturns":
        table = IndustryReturns
    elif table_name == "PValueData":
        table = PValueData
    elif table_name == "ResTest":
        table = ResTest
    elif table_name == "SE":
        table = SE
    elif table_name == "VifValues":
        table = VifValues
    elif table_name == "XDict":
        table = XDict
    else: 
        raise HTTPException(status_code=404, detail="Table not found")
    data = get_model_data_by_date(db, table, start_date, end_date)
    return data

