import os
from sqlalchemy import create_engine, text

DB_URL = os.getenv("DB_URL", "postgresql+psycopg://postgres:postgres@localhost:5442/legalmind")
engine = create_engine(DB_URL, future=True)

def ping_db() -> bool:
    with engine.connect() as conn:
        return conn.execute(text("select 1")).scalar_one() == 1
