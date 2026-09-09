import os
from fastapi import Security, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from dotenv import load_dotenv

load_dotenv()
API_SECRET_KEY = os.getenv("API_SECRET_KEY", "token-secreto-compartilhado-123")

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