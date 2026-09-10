from fastapi import FastAPI
from fastapi.security import HTTPBearer
import os
from dotenv import load_dotenv
from fastapi import FastAPI, Depends

from api.controllers.pet_controller import router as pet_router

from fastapi import Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from json import JSONDecodeError

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

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    erros_personalizados = []
    for error in exc.errors():
        campo = " -> ".join(str(x) for x in error["loc"])
        msg_original = error["msg"]
        
        if "greater than" in msg_original:
            msg = "O valor deve ser maior do que o limite permitido."
        elif "less than" in msg_original:
            msg = "O valor deve ser menor do que o limite permitido."
        elif "Field required" in msg_original:
            msg = "Este campo é obrigatório."
        else:
            msg = msg_original
            
        erros_personalizados.append({"campo": campo, "erro": msg})
        
    return JSONResponse(
        status_code=422,
        content={"detail": erros_personalizados, "mensagem": "Erro de validação nos dados enviados."}
    )
    

@app.exception_handler(JSONDecodeError)
async def json_decode_exception_handler(request: Request, exc: JSONDecodeError):
    return JSONResponse(
        status_code=422,
        content={
            "detail": [{"campo": "body", "erro": "Tipo inválido. O formato do JSON enviado é inválido ou contém tipos numéricos incorretos."}],
            "mensagem": "Erro de validação nos dados enviados."
        }
    )