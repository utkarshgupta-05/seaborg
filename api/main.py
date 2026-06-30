import os

from dotenv import load_dotenv
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import text

logger = logging.getLogger(__name__)

from api.routes import chat, data, export
from api.database import get_engine
from rag.retriever import load_index

load_dotenv()

# ── Startup ───────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Runs at server startup.

    Loads the FAISS index into memory and verifies the database connection.
    Prints 'SeaBorg API ready.' on success.

    Side effects:
        Loads FAISS index and Parquet DataFrame into module-level rag.retriever state.
        Opens and closes a PostgreSQL connection to verify connectivity.
    """
    load_index()

    engine = get_engine()
    with engine.connect() as conn:
        conn.execute(text("SELECT 1"))

    logger.info("SeaBorg API ready.")
    yield

app = FastAPI(title="SeaBorg API", version="1.0.0", lifespan=lifespan)

# ── CORS ──────────────────────────────────────────────────────────────────────
_environment = os.getenv("ENVIRONMENT", "development")
if _environment == "production":
    _origins = [os.getenv("FRONTEND_URL", "")]
else:
    _origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routers ───────────────────────────────────────────────────────────────────
app.include_router(chat.router, prefix="/api")
app.include_router(data.router, prefix="/api")
app.include_router(export.router, prefix="/api")

@app.get("/")
async def root():
    return {"status": "ok", "service": "SeaBorg API"}

