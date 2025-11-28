from datetime import datetime
from sqlalchemy import Column, Integer, String, DateTime, Text
from database import Base


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)

    # Apple-User-ID (sub aus dem Apple-Token)
    user_id = Column(String, unique=True, index=True, nullable=False)

    # Abo-Plan: free / pro / premium
    plan = Column(String, default="free", nullable=False)

    # Wie viele KI-Anfragen in diesem Monat schon gemacht wurden
    monthly_usage = Column(Integer, default=0, nullable=False)

    # Welcher Monat gezählt wird (Format "YYYY-MM", z.B. "2025-11")
    usage_month = Column(String, nullable=True)

    # Liste erledigter/ausgeschlossener Fälle als JSON-Text (z.B. '["UUID1","UUID2"]')
    excluded_cases = Column(Text, nullable=True)

    # NEU: Notizen pro Fall als JSON-Text (z.B. '{"caseId":"Meine Notiz"}')
    notes = Column(Text, nullable=True)

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(
        DateTime,
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
        nullable=False,
    )
