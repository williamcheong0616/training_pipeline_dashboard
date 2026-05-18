import os
from pathlib import Path
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from .models import Base

def _default_db_url() -> str:
    docker_dir = Path("/app/data")
    if docker_dir.exists():
        return f"sqlite:///{docker_dir}/forge.db"
    local_dir = Path(__file__).resolve().parents[2] / "data"
    local_dir.mkdir(exist_ok=True)
    return f"sqlite:///{local_dir}/forge.db"

DATABASE_URL = os.getenv("DATABASE_URL", _default_db_url())

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {},
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def init_db():
    Base.metadata.create_all(bind=engine)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
