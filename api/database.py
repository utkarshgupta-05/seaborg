import os
from sqlalchemy import create_engine

_engine = None

def get_engine():
    """Returns a shared SQLAlchemy engine, creating it on first call."""
    global _engine
    if _engine is None:
        db_url = os.getenv("DATABASE_URL")
        if not db_url:
            raise ValueError("DATABASE_URL is not set.")
        _engine = create_engine(db_url, future=True, pool_pre_ping=True)
    return _engine
