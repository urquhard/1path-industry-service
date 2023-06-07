from fastapi import FastAPI
from finance import finance_router
from model import model_router
#from settings import settings

#app = FastAPI(title="Industry Api", openapi_url=settings.openapi_url)
app = FastAPI(title="Industry Api")

app.include_router(prefix="/api/finance", tags=['Endpoints'], router=finance_router)
app.include_router(prefix="/api/model", tags=['Endpoints'], router=model_router)
