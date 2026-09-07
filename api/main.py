from fastapi import FastAPI
import os
from dotenv import load_dotenv

from api.controllers.pet_controller import router as pet_router

load_dotenv()

ENV = os.getenv("APP_ENV", "development")
SECRET_KEY = os.getenv("API_SECRET_KEY", "default-secret")

app = FastAPI(
    title="Vitalia AI API", 
    description="API de Inteligência Artificial para Pets com LLM", 
    version="1.2"
)

# Inclui todas as rotas de pets centralizadas no controller
app.include_router(pet_router)