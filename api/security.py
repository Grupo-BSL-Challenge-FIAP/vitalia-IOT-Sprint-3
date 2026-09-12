import os
from fastapi import Security, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from dotenv import load_dotenv

load_dotenv()

API_SECRET_KEY = os.getenv("API_SECRET_KEY")
if not API_SECRET_KEY:
    raise ValueError("A variável de ambiente API_SECRET_KEY não está definida.")

security = HTTPBearer()

def verificar_autenticacao(credentials: HTTPAuthorizationCredentials = Security(security)):
    token = credentials.credentials
    if token != API_SECRET_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Credenciais de autenticação inválidas ou token incorreto.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return token