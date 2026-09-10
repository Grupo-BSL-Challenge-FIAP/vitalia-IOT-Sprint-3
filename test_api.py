from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)

response = client.get("/insights/1000")
print("Status Code:", response.status_code)
print("Resposta Insights:", response.json())