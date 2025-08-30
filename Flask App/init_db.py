# scripts/init_db.py
import os
from models import init_db

print("Using DB:", os.environ.get("DATABASE_URL", "sqlite:///weeklybuddy.db"))
init_db()
print("✅ Tables ensured.")
