from datetime import datetime
from sqlalchemy import Column, Integer, String, DateTime
from database import Base


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String, unique=True, index=True, nullable=False)  # von Apple
    plan = Column(String, default="free", nullable=False)              # free / pro / premium
    monthly_usage = Column(Integer, default=0, nullable=False)
    usage_month = Column(String, nullable=True)  # Format "YYYY-MM"

    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(
        DateTime,
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
        nullable=False,
    )

