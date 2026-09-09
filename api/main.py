from fastapi import FastAPI
from fastapi.security import HTTPBearer
import os
from dotenv import load_dotenv
from fastapi import FastAPI, Depends

from api.controllers.pet_controller import router as pet_router

load_dotenv()

ENV = os.getenv("APP_ENV", "development")
SECRET_KEY = os.getenv("API_SECRET_KEY", "default-secret")

security = HTTPBearer()

app = FastAPI(
    title="Vitalia AI API", 
    description="API de Inteligência Artificial para Pets com LLM", 
    version="1.2",
    swagger_ui_parameters={"persistAuthorization": True},
    dependencies=[Depends(security)]
)

app.include_router(pet_router)