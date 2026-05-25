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
    # Additive migrations for SQLite (safe to re-run — silently ignored if column exists)
    if DATABASE_URL.startswith("sqlite"):
        from sqlalchemy import text
        with engine.connect() as conn:
            for stmt in [
                "ALTER TABLE jobs ADD COLUMN remarks TEXT",
                "ALTER TABLE datasets ADD COLUMN template TEXT",
                "CREATE INDEX IF NOT EXISTS ix_training_metrics_job_id ON training_metrics (job_id)",
                "ALTER TABLE conversations ADD COLUMN gen_params JSON",
            ]:
                try:
                    conn.execute(text(stmt))
                    conn.commit()
                except Exception:
                    pass


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
