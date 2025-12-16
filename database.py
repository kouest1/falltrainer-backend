import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

# DATABASE_URL kommt von Render (Environment Variable)
DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL ist nicht gesetzt")

# Render gibt oft postgres:// aus, SQLAlchemy will i.d.R. postgresql://
# (und mit psycopg2 Treiber am saubersten: postgresql+psycopg2://)
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql+psycopg2://", 1)

# Engine-Konfiguration
# - pool_pre_ping: erkennt tote Connections (wichtig bei Render/Cloud)
# - pool_recycle: vermeidet stale Connections (optional)
# - pool_size/max_overflow: moderat, damit Render nicht "überläuft"
POOL_SIZE = int(os.getenv("DB_POOL_SIZE", "5"))
MAX_OVERFLOW = int(os.getenv("DB_MAX_OVERFLOW", "10"))
POOL_RECYCLE = int(os.getenv("DB_POOL_RECYCLE", "1800"))  # Sekunden

if DATABASE_URL.startswith("sqlite"):
    # Falls du lokal mal SQLite nutzt
    engine = create_engine(
        DATABASE_URL,
        connect_args={"check_same_thread": False},
        future=True,
    )
else:
    engine = create_engine(
        DATABASE_URL,
        pool_pre_ping=True,
        pool_recycle=POOL_RECYCLE,
        pool_size=POOL_SIZE,
        max_overflow=MAX_OVERFLOW,
        future=True,
    )

SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine,
    expire_on_commit=False,  # praktisch für API-Responses nach commit
)

Base = declarative_base()

# Optionaler Helper (falls du ihn mal direkt aus database.py importieren willst)
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

