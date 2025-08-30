# scripts/add_archived_at.py
import os
from sqlalchemy import create_engine, text

DB_URL = os.environ.get("DATABASE_URL", "sqlite:///weeklybuddy.db")
engine = create_engine(
    DB_URL,
    connect_args={"check_same_thread": False} if DB_URL.startswith("sqlite") else {}
)

with engine.begin() as conn:
    dialect = engine.dialect.name
    if dialect == "sqlite":
        cols = [row["name"] for row in conn.execute(text("PRAGMA table_info(responses)")).mappings()]
        if "archived_at" not in cols:
            conn.execute(text("ALTER TABLE responses ADD COLUMN archived_at DATETIME"))
            print("✅ Added archived_at to SQLite responses table.")
        else:
            print("ℹ️ archived_at already exists.")
    elif dialect in ("postgresql", "postgres"):
        conn.execute(text("ALTER TABLE responses ADD COLUMN IF NOT EXISTS archived_at TIMESTAMP NULL"))
        conn.execute(text("CREATE INDEX IF NOT EXISTS ix_responses_archived_at ON responses(archived_at)"))
        print("✅ Ensured archived_at exists on Postgres.")
    else:
        raise SystemExit(f"Unsupported dialect: {dialect}")
