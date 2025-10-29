from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="LexMind API v0.3")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get('/')
def root():
    return {
        'service': 'LexMind API',
        'status': 'running',
        'version': '0.3.0',
        'message': 'Backend fonctionnel sans DB pour test'
    }

@app.get('/health')
def health():
    return {'status': 'healthy'}

@app.get('/api/info')
def info():
    return {
        'features': ['health-check', 'basic-api'],
        'ready_for': 'MCP integration'
    }
