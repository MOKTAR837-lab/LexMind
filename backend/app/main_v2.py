from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import text
from app.db import ping_db, engine
from typing import Optional
import os

app = FastAPI(title="LexMind API")

# Configuration
ML_ENABLED = os.getenv("ML_ENABLED", "false").lower() == "true"

# ChatHandler sera chargé seulement si ML activé
chat_handler = None

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Schéma DB au startup ---
@app.on_event("startup")
def ensure_schema():
    with engine.begin() as conn:
        conn.execute(text('''
        CREATE TABLE IF NOT EXISTS folders (
          id BIGSERIAL PRIMARY KEY,
          name TEXT NOT NULL,
          parent_id BIGINT REFERENCES folders(id) ON DELETE SET NULL,
          created_at TIMESTAMPTZ DEFAULT now()
        );
        '''))
        conn.execute(text('''
        CREATE TABLE IF NOT EXISTS documents (
          id BIGSERIAL PRIMARY KEY,
          filename TEXT NOT NULL,
          content TEXT,
          folder_id BIGINT REFERENCES folders(id) ON DELETE SET NULL,
          created_at TIMESTAMPTZ DEFAULT now()
        );
        '''))
        conn.execute(text('CREATE INDEX IF NOT EXISTS idx_documents_folder_id ON documents(folder_id);'))
    
    # Charger ChatHandler seulement si ML activé
    if ML_ENABLED:
        global chat_handler
        from app.chat_handler import ChatHandler
        chat_handler = ChatHandler()
        print('✅ ChatHandler chargé avec ML')
    else:
        print('⚠️  ML désactivé - mode light')

# --- Health checks ---
@app.get('/')
def root():
    return {
        'service': 'LexMind API',
        'status': 'running',
        'ml_enabled': ML_ENABLED,
        'version': '0.2.0'
    }

@app.get('/health')
def health():
    return {'status': 'healthy', 'ml_enabled': ML_ENABLED}

@app.get('/db-check')
def db_check():
    return {'ok': ping_db()}

# --- Endpoints Documents ---
@app.post('/api/upload/document')
async def upload_document(
    file: UploadFile = File(...),
    folder_id: Optional[int] = None
):
    '''Upload un document'''
    if not file.filename:
        raise HTTPException(400, 'Nom de fichier requis')
    
    # Lire contenu
    content = await file.read()
    text_content = content.decode('utf-8', errors='ignore')
    
    # Sauvegarder en DB
    with engine.begin() as conn:
        result = conn.execute(
            text('''
            INSERT INTO documents (filename, content, folder_id)
            VALUES (:filename, :content, :folder_id)
            RETURNING id
            '''),
            {'filename': file.filename, 'content': text_content, 'folder_id': folder_id}
        )
        doc_id = result.fetchone()[0]
    
    # Indexer si ML activé
    if ML_ENABLED and chat_handler:
        # TODO: indexer dans Qdrant
        pass
    
    return {
        'id': doc_id,
        'filename': file.filename,
        'indexed': ML_ENABLED
    }

@app.get('/api/documents')
def list_documents():
    '''Liste tous les documents'''
    with engine.begin() as conn:
        result = conn.execute(text('SELECT id, filename, created_at FROM documents ORDER BY created_at DESC'))
        docs = [{'id': r[0], 'filename': r[1], 'created_at': str(r[2])} for r in result]
    return {'documents': docs, 'count': len(docs)}

# --- Endpoint Chat (si ML activé) ---
@app.post('/api/chat')
async def chat(query: str, limit: int = 5):
    '''Recherche et chat'''
    if not ML_ENABLED or not chat_handler:
        raise HTTPException(503, 'ML non activé')
    
    # TODO: utiliser chat_handler
    return {'message': 'ML activé mais pas encore implémenté', 'query': query}

# --- WebSocket (si ML activé) ---
@app.websocket('/ws/chat/{user_id}')
async def ws_chat(websocket: WebSocket, user_id: str):
    if not ML_ENABLED:
        await websocket.close(code=1003, reason='ML non activé')
        return
    
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()
            # TODO: traiter avec chat_handler
            await websocket.send_json({'response': f'Reçu: {data}'})
    except WebSocketDisconnect:
        pass

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=8000)
