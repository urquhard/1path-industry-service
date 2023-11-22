from fastapi import FastAPI
from finance import finance_router
from model import model_router
from fastapi.middleware.cors import CORSMiddleware

#from settings import settings

#app = FastAPI(title="Industry Api", openapi_url=settings.openapi_url)
app = FastAPI(title="Industry Api")
origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(prefix="/api/finance", tags=['Endpoints'], router=finance_router)
app.include_router(prefix="/api/model", tags=['Endpoints'], router=model_router)
