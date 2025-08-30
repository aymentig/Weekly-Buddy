# models.py  (REPLACE the whole file with this)

from sqlalchemy import Column, Integer, String, Text, DateTime, Boolean, ForeignKey, create_engine
from sqlalchemy.orm import declarative_base, sessionmaker, relationship
from sqlalchemy.sql import func
import os

DATABASE_URL = os.environ.get("DATABASE_URL", "sqlite:///weeklybuddy.db")

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Existing models you already had:
class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True)
    slack_id = Column(String(64), unique=True, index=True)
    name = Column(String(128))
    responses = relationship("Response", back_populates="user")

class Response(Base):
    __tablename__ = "responses"
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"), index=True)
    worked_on = Column(Text, default="No response")
    next_up = Column(Text, default="No response")
    blockers = Column(Text, default="No response")
    created_at = Column(DateTime, server_default=func.now())  # ensures dates moving forward
    # NEW: soft-archive flag
    archived_at = Column(DateTime, nullable=True, index=True)

    user = relationship("User", back_populates="responses")

# NEW: Tagging
class Tag(Base):
    __tablename__ = "tags"
    id = Column(Integer, primary_key=True)
    name = Column(String(64), unique=True, index=True)

class ResponseTag(Base):
    __tablename__ = "response_tags"
    id = Column(Integer, primary_key=True)
    response_id = Column(Integer, ForeignKey("responses.id"), index=True)
    tag_id = Column(Integer, ForeignKey("tags.id"), nullable=True)
    tagged_user_id = Column(Integer, ForeignKey("users.id"), nullable=True)

def init_db():
    Base.metadata.create_all(bind=engine)
