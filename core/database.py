from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

SQLALCHEMY_DATABASE_URL = "sqlite:///./saas_data.db"

engine = create_engine(
    SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False}
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

def _ensure_sqlite_columns():
    try:
        from sqlalchemy import inspect, text
        inspector = inspect(engine)
        if "bots" in inspector.get_table_names():
            columns = [col["name"] for col in inspector.get_columns("bots")]
            if "pos_endpoints" not in columns:
                with engine.connect() as conn:
                    conn.execute(text("ALTER TABLE bots ADD COLUMN pos_endpoints TEXT DEFAULT '[]'"))
                    conn.commit()
    except Exception as e:
        print(f"[DB] Advertencia al verificar columna pos_endpoints: {e}")

_ensure_sqlite_columns()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
